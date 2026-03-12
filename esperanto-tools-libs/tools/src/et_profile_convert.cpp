// et-profile-convert: Convert Esperanto runtime profiler traces (cereal JSON)
// to Chrome Trace Event Format for visualization in chrome://tracing or Perfetto.
//
// Input:  ET runtime trace JSON -- top-level object {"value0": <Event>, ...}
//         where each event has type, class, timeStamp, thread_id, and extra fields.
//         The "extra" array uses cereal's variant encoding (index + data).
//
// Output: Chrome Trace Event Format JSON with:
//         - Host-side events on pid=0 (mapped from ET event types)
//         - Synthetic device-side execution spans on pid=1 (from ResponseReceived data)
//         - Thread/process name metadata events
//         - Memory counter events
//
// Device clock sync: The device and host run independent clocks. We compute
// the lower convex hull of (device_time, host_time) points from ResponseReceived
// events, then fit a linear model (scale + offset) from the hull endpoints to
// correct for both clock offset and drift.
//
// Copyright (c) 2025 Ainekko, Co.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

// -- Utilities --

template <typename T>
std::optional<T> parse_number(std::string_view s) {
    if (s.empty()) return std::nullopt;
    T val{};
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
    if (ec == std::errc{} && ptr == s.data() + s.size())
        return val;
    return std::nullopt;
}

// Safely extract a value from a JSON node, returning nullopt on any error.
template <typename T>
std::optional<T> try_get(const json& j) {
    try {
        return j.get<T>();
    } catch (...) {
        return std::nullopt;
    }
}

// -- Input types (cereal JSON format) --
//
// Each ET event carries its data in an "extra" array of {key, value} pairs.
// The value is a cereal variant: {index: N, data: ...} where the data shape
// depends on the index (see extract_extras for the full mapping).

struct TimeCount {
    int64_t count{};
};

void from_json(const json& j, TimeCount& tc) {
    j.at("count").get_to(tc.count);
}

struct TimeSinceEpoch {
    TimeCount time_since_epoch{};
};

void from_json(const json& j, TimeSinceEpoch& tse) {
    j.at("time_since_epoch").get_to(tse.time_since_epoch);
}

struct ExtraVariant {
    int index{};
    json data{};
};

void from_json(const json& j, ExtraVariant& ev) {
    j.at("index").get_to(ev.index);
    ev.data = j.at("data");
}

struct ExtraEntry {
    std::string key{};
    ExtraVariant value{};
};

void from_json(const json& j, ExtraEntry& ee) {
    j.at("key").get_to(ee.key);
    j.at("value").get_to(ee.value);
}

struct EtEvent {
    std::string type{};
    std::string class_{};
    TimeSinceEpoch timeStamp{};
    std::string thread_id{};
    std::vector<ExtraEntry> extra{};
};

// "class" is a C++ keyword -- map from JSON key "class" to class_
void from_json(const json& j, EtEvent& ev) {
    j.at("type").get_to(ev.type);
    j.at("class").get_to(ev.class_);
    j.at("timeStamp").get_to(ev.timeStamp);
    j.at("thread_id").get_to(ev.thread_id);
    j.at("extra").get_to(ev.extra);
}

// -- Extracted extras --
//
// Flattened view of the extra array for convenient access.
// Fields are optional -- only populated if present in the event.

struct ExtractedExtras {
    std::optional<int64_t> duration_ns;
    std::optional<uint64_t> event_id;
    std::optional<uint64_t> parent_id;
    std::optional<int> stream_id;
    std::optional<int> device_id;
    std::optional<int> kernel_id;
    std::optional<int> rsp_type;
    std::optional<uint64_t> device_cmd_exec_dur;
    std::optional<uint64_t> device_cmd_wait_dur;
    std::optional<uint64_t> device_cmd_start_ts;
    std::optional<uint64_t> size;
    std::optional<std::string> thread_name;
    std::optional<uint32_t> frequency;
    std::optional<uint64_t> allocated_mem;
    std::optional<uint64_t> free_mem;
    std::optional<uint64_t> max_contig_free;
    std::optional<uint64_t> ptr;
    std::optional<uint64_t> src_ptr;
    std::optional<uint64_t> dst_ptr;
    std::optional<uint64_t> load_address;
    std::optional<uint32_t> alignment;
    std::optional<bool> barrier;
};

// Extract known fields from an event's extra array.
//
// Cereal variant index mapping:
//   0: uint64  -- plain number (size, pointer, device timing, memory stats)
//   1: uint64  -- event/parent id
//   2: int     -- stream id
//   3: int     -- device id
//   4: int     -- kernel id
//   5: int     -- response type
//   6: object  -- Duration {count: N} in nanoseconds
//   7: object  -- DeviceProperties (large struct, we extract frequency only)
//   8: uint64  -- version number
//  10: string  -- thread name
//  11: bool    -- barrier flag
//  12: uint32  -- alignment
ExtractedExtras extract_extras(const std::vector<ExtraEntry>& extras) {
    ExtractedExtras ex;
    for (auto& e : extras) {
        const auto& key = e.key;
        const auto& data = e.value.data;
        int idx = e.value.index;

        if (key == "duration" && idx == 6) {
            if (auto tc = try_get<TimeCount>(data)) ex.duration_ns = tc->count;
        } else if (key == "event" && idx == 1) {
            ex.event_id = try_get<uint64_t>(data);
        } else if (key == "parent_id" && idx == 1) {
            ex.parent_id = try_get<uint64_t>(data);
        } else if (key == "stream" && idx == 2) {
            ex.stream_id = try_get<int>(data);
        } else if (key == "device_id" && idx == 3) {
            ex.device_id = try_get<int>(data);
        } else if (key == "kernel_id" && idx == 4) {
            ex.kernel_id = try_get<int>(data);
        } else if (key == "rsp_type" && idx == 5) {
            ex.rsp_type = try_get<int>(data);
        } else if (key == "device_properties" && idx == 7) {
            // Lenient extraction -- only grab frequency from the large object
            try {
                if (data.contains("frequency"))
                    ex.frequency = data.at("frequency").get<uint32_t>();
            } catch (...) {}
        } else if (key == "thread_name" && idx == 10) {
            ex.thread_name = try_get<std::string>(data);
        } else if (key == "barrier" && idx == 11) {
            ex.barrier = try_get<bool>(data);
        } else if (key == "alignment" && idx == 12) {
            ex.alignment = try_get<uint32_t>(data);
        } else if (idx == 0) {
            if (auto val = try_get<uint64_t>(data)) {
                if (key == "device_cmd_exec_dur") ex.device_cmd_exec_dur = val;
                else if (key == "device_cmd_wait_dur") ex.device_cmd_wait_dur = val;
                else if (key == "device_cmd_start_ts") ex.device_cmd_start_ts = val;
                else if (key == "size") ex.size = val;
                else if (key == "mem.allocated_memory") ex.allocated_mem = val;
                else if (key == "mem.free_memory") ex.free_mem = val;
                else if (key == "mem.max_contiguous_free_mem") ex.max_contig_free = val;
                else if (key == "ptr") ex.ptr = val;
                else if (key == "src_ptr") ex.src_ptr = val;
                else if (key == "dst_ptr") ex.dst_ptr = val;
                else if (key == "load_address") ex.load_address = val;
            }
        }
    }
    return ex;
}

// -- Category mapping --
// Groups ET event classes into Chrome trace categories for filtering.

std::string_view category_for(std::string_view cls) {
    if (cls == "KernelLaunch" || cls == "LoadCode" || cls == "UnloadCode")
        return "kernel";
    if (cls == "MallocDevice" || cls == "FreeDevice" ||
        cls == "MemcpyHostToDevice" || cls == "MemcpyDeviceToHost" ||
        cls == "MemcpyDeviceToDevice" || cls == "CmaCopy" || cls == "MemoryStats")
        return "memory";
    if (cls == "WaitForEvent" || cls == "WaitForStream")
        return "sync";
    if (cls == "CommandSent" || cls == "ResponseReceived" ||
        cls == "DispatchEvent" || cls == "DeviceCommand")
        return "command";
    return "system";
}

// -- Output types (Chrome Trace Event Format) --

struct ChromeArgs {
    std::optional<uint64_t> event_id;
    std::optional<uint64_t> parent_id;
    std::optional<int> stream_id;
    std::optional<int> device_id;
    std::optional<int> kernel_id;
    std::optional<uint64_t> size;
    std::optional<uint64_t> ptr;
    std::optional<uint64_t> src_ptr;
    std::optional<uint64_t> dst_ptr;
    std::optional<uint64_t> load_address;
    std::optional<uint32_t> alignment;
    std::optional<bool> barrier;
    std::optional<int> rsp_type;
    std::optional<uint64_t> allocated;
    std::optional<uint64_t> free;
    std::optional<uint64_t> max_contiguous_free;
    std::optional<int64_t> wait_ns;
    std::optional<std::string> name;
};

// Only emit fields that are present (skip nullopt)
void to_json(json& j, const ChromeArgs& a) {
    j = json::object();
    if (a.event_id) j["event_id"] = *a.event_id;
    if (a.parent_id) j["parent_id"] = *a.parent_id;
    if (a.stream_id) j["stream_id"] = *a.stream_id;
    if (a.device_id) j["device_id"] = *a.device_id;
    if (a.kernel_id) j["kernel_id"] = *a.kernel_id;
    if (a.size) j["size"] = *a.size;
    if (a.ptr) j["ptr"] = *a.ptr;
    if (a.src_ptr) j["src_ptr"] = *a.src_ptr;
    if (a.dst_ptr) j["dst_ptr"] = *a.dst_ptr;
    if (a.load_address) j["load_address"] = *a.load_address;
    if (a.alignment) j["alignment"] = *a.alignment;
    if (a.barrier) j["barrier"] = *a.barrier;
    if (a.rsp_type) j["rsp_type"] = *a.rsp_type;
    if (a.allocated) j["allocated"] = *a.allocated;
    if (a.free) j["free"] = *a.free;
    if (a.max_contiguous_free) j["max_contiguous_free"] = *a.max_contiguous_free;
    if (a.wait_ns) j["wait_ns"] = *a.wait_ns;
    if (a.name) j["name"] = *a.name;
}

struct ChromeEvent {
    std::string ph;
    std::string name;
    std::optional<std::string> cat;
    int64_t ts{};
    std::optional<int64_t> dur;
    int pid{};
    uint64_t tid{};
    std::optional<std::string> s;
    std::optional<ChromeArgs> args;
};

void to_json(json& j, const ChromeEvent& e) {
    j = json{
        {"ph", e.ph},
        {"name", e.name},
        {"ts", e.ts},
        {"pid", e.pid},
        {"tid", e.tid}
    };
    if (e.cat) j["cat"] = *e.cat;
    if (e.dur) j["dur"] = *e.dur;
    if (e.s) j["s"] = *e.s;
    if (e.args) j["args"] = *e.args;
}

struct ChromeTrace {
    std::string displayTimeUnit = "ns";
    std::vector<ChromeEvent> traceEvents;
};

void to_json(json& j, const ChromeTrace& t) {
    j = json{
        {"displayTimeUnit", t.displayTimeUnit},
        {"traceEvents", t.traceEvents}
    };
}

// -- File I/O helper --

std::string read_file(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs)
        throw std::runtime_error("cannot open " + path.string());
    auto sz = ifs.tellg();
    if (sz == -1)
        throw std::runtime_error("failed to get size of " + path.string());
    ifs.seekg(0);
    std::string buf(static_cast<size_t>(sz), '\0');
    if (!ifs.read(buf.data(), sz))
        throw std::runtime_error("failed to read " + path.string());
    return buf;
}

// -- CLI --

void print_usage() {
    std::cout <<
        "et-profile-convert -- Convert ET runtime traces to Chrome Trace Format\n"
        "\n"
        "Usage:\n"
        "  et-profile-convert <input.json> [output.json] [-k kernel_id.json]\n"
        "\n"
        "Arguments:\n"
        "  input.json         ET runtime profiler trace (cereal JSON format)\n"
        "  output.json        Output path (default: <input>_chrome.json)\n"
        "\n"
        "Options:\n"
        "  -k <kernel_id.json>  Kernel name mapping ({\"name\": id, ...})\n"
        "                       Auto-detected as kernel_id.json next to input if present\n"
        "  -h, --help           Show this help message\n"
        "\n"
        "Output can be loaded in chrome://tracing or https://ui.perfetto.dev\n";
}

struct Args {
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path kernel_ids;
};

// Returns: 0 = parsed ok, 1 = error, 2 = help printed (exit 0)
int parse_args(int argc, char* argv[], Args& args) {
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage();
            return 2;
        }
        if (arg == "-k") {
            if (++i >= argc) {
                std::cerr << "Error: -k requires an argument\n";
                return 1;
            }
            args.kernel_ids = argv[i];
            continue;
        }
        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Error: unknown option '" << arg << "'\n";
            return 1;
        }
        positional.emplace_back(arg);
    }

    if (positional.empty()) {
        std::cerr << "Error: missing input file\n\n";
        print_usage();
        return 1;
    }
    if (positional.size() > 2) {
        std::cerr << "Error: too many arguments\n\n";
        print_usage();
        return 1;
    }

    args.input = positional[0];
    if (positional.size() >= 2) {
        args.output = positional[1];
    } else {
        args.output = args.input.parent_path() / (args.input.stem().string() + "_chrome.json");
    }

    return 0;
}

// -- Main --

int main(int argc, char* argv[]) {
    Args args;
    if (int rc = parse_args(argc, argv, args); rc != 0)
        return rc == 2 ? 0 : rc;

    auto& [input_path, output_path, kernel_id_path] = args;

    // Load optional kernel name->id mapping and invert to id->name.
    std::unordered_map<int, std::string> kernel_names;
    {
        bool explicit_k = !kernel_id_path.empty();
        auto kid_path = explicit_k
            ? kernel_id_path
            : input_path.parent_path() / "kernel_id.json";

        if (explicit_k && !std::filesystem::exists(kid_path)) {
            std::cerr << "Error: kernel id file not found: " << kid_path.string() << "\n";
            return 1;
        }

        if (std::filesystem::exists(kid_path)) {
            try {
                auto kid_buf = read_file(kid_path);
                auto kid_json = json::parse(kid_buf);
                auto name_to_id = kid_json.get<std::map<std::string, int>>();
                for (auto& [name, id] : name_to_id)
                    kernel_names[id] = name;
                std::cout << "Loaded " << kernel_names.size() << " kernel names from " << kid_path.string() << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error: failed to parse " << kid_path.string() << ": " << e.what() << "\n";
                return 1;
            }
        }
    }

    // Read and parse input trace
    std::cout << "Reading " << input_path.string() << "...\n";
    auto input_buf = read_file(input_path);
    std::cout << "Parsing " << input_buf.size() << " bytes...\n";

    std::map<std::string, EtEvent> events;
    try {
        auto input_json = json::parse(input_buf);
        events = input_json.get<std::map<std::string, EtEvent>>();
    } catch (const std::exception& e) {
        std::cerr << "Error: failed to parse " << input_path.string() << ": " << e.what() << "\n";
        return 1;
    }
    input_buf.clear();
    input_buf.shrink_to_fit();
    std::cout << "Parsed " << events.size() << " events\n";

    // -- First pass: collect metadata --
    uint32_t device_freq = 0;
    std::unordered_map<uint64_t, std::string> thread_names;
    std::unordered_map<uint64_t, int> event_to_kernel;
    for (auto& [k, ev] : events) {
        if (ev.class_ == "GetDeviceProperties") {
            auto ex = extract_extras(ev.extra);
            if (ex.frequency) device_freq = *ex.frequency;
        } else if (ev.class_ == "IdentifyThread") {
            auto ex = extract_extras(ev.extra);
            if (ex.thread_name) {
                if (auto tid = parse_number<uint64_t>(ev.thread_id))
                    thread_names[*tid] = *ex.thread_name;
            }
        } else if (ev.class_ == "KernelLaunch") {
            auto ex = extract_extras(ev.extra);
            if (ex.event_id && ex.kernel_id)
                event_to_kernel[*ex.event_id] = *ex.kernel_id;
        }
    }

    // -- Device-to-host clock synchronization --
    double clock_scale = 1.0;
    double clock_offset = 0.0;
    if (device_freq > 0) {
        struct ClockSample { int64_t dev_us; int64_t host_us; };
        std::vector<ClockSample> samples;
        for (auto& [k, ev] : events) {
            if (ev.class_ != "ResponseReceived") continue;
            auto ex = extract_extras(ev.extra);
            if (!ex.device_cmd_start_ts || !ex.device_cmd_exec_dur) continue;
            int64_t wait_ticks = ex.device_cmd_wait_dur.value_or(0);
            int64_t dev_end_ticks = static_cast<int64_t>(*ex.device_cmd_start_ts)
                                  + wait_ticks
                                  + static_cast<int64_t>(*ex.device_cmd_exec_dur);
            int64_t dev_us = dev_end_ticks * 1000 / device_freq / 1000;
            int64_t host_us = ev.timeStamp.time_since_epoch.count / 1000;
            samples.push_back({dev_us, host_us});
        }

        if (samples.size() >= 2) {
            std::sort(samples.begin(), samples.end(),
                      [](auto& a, auto& b) { return a.dev_us < b.dev_us; });

            // Lower Convex Hull (Monotone Chain) -- finds the bottom envelope
            // of (device, host) points, filtering for lowest-latency samples.
            std::vector<ClockSample> lower_hull;
            for (const auto& p : samples) {
                while (lower_hull.size() >= 2) {
                    const auto& a = lower_hull[lower_hull.size() - 2];
                    const auto& b = lower_hull.back();
                    int64_t cross = (b.dev_us - a.dev_us) * (p.host_us - a.host_us) -
                                    (b.host_us - a.host_us) * (p.dev_us - a.dev_us);
                    if (cross >= 0) lower_hull.pop_back();
                    else break;
                }
                lower_hull.push_back(p);
            }

            if (lower_hull.size() >= 2) {
                auto s0 = lower_hull.front();
                auto s1 = lower_hull.back();

                if (s1.dev_us != s0.dev_us) {
                    clock_scale = static_cast<double>(s1.host_us - s0.host_us)
                                / static_cast<double>(s1.dev_us - s0.dev_us);
                    clock_offset = s0.host_us - clock_scale * s0.dev_us;
                }
            }

            double drift_ppm = (clock_scale - 1.0) * 1e6;
            std::cout << "Clock sync: " << samples.size() << " samples ("
                      << lower_hull.size() << " on lower hull), scale="
                      << std::setprecision(9) << std::fixed << clock_scale
                      << " (" << std::setprecision(1) << std::showpos << drift_ppm
                      << std::noshowpos << " ppm), offset="
                      << std::setprecision(0) << clock_offset << " us\n"
                      << std::defaultfloat;
        } else {
            std::cerr << "Warning: insufficient clock sync samples (" << samples.size()
                      << "), device timestamps may be inaccurate\n";
        }
    }

    // Convert device clock ticks to host-aligned microseconds
    auto dev_to_host_us = [&](int64_t dev_ticks) -> std::optional<int64_t> {
        if (device_freq == 0) return std::nullopt;
        int64_t dev_us = dev_ticks * 1000 / device_freq / 1000;
        return static_cast<int64_t>(clock_scale * dev_us + clock_offset);
    };

    std::cout << "Device frequency: " << device_freq << " MHz, " << thread_names.size() << " named threads\n";

    // -- Second pass: build Chrome trace events --
    ChromeTrace trace;
    trace.traceEvents.reserve(events.size() + thread_names.size() + 4);

    // Process and thread name metadata (ph=M)
    trace.traceEvents.push_back({
        "M", "process_name", std::nullopt, 0, std::nullopt, 0, 0, std::nullopt,
        ChromeArgs{std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::string{"Host"}}
    });
    trace.traceEvents.push_back({
        "M", "process_name", std::nullopt, 0, std::nullopt, 1, 0, std::nullopt,
        ChromeArgs{std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
                   std::nullopt, std::nullopt, std::string{"Device"}}
    });
    for (auto& [tid, tname] : thread_names) {
        ChromeArgs targs;
        targs.name = tname;
        trace.traceEvents.push_back({
            "M", "thread_name", std::nullopt, 0, std::nullopt, 0, tid, std::nullopt,
            std::move(targs)
        });
    }

    for (auto& [k, ev] : events) {
        auto ex = extract_extras(ev.extra);
        auto tid_opt = parse_number<uint64_t>(ev.thread_id);
        if (!tid_opt) continue;
        uint64_t tid = *tid_opt;
        int64_t ts = ev.timeStamp.time_since_epoch.count / 1000; // ns -> us

        // Map ET event type -> Chrome trace phase
        std::string ph;
        if (ev.type == "Complete") ph = "X";
        else if (ev.type == "Start") ph = "B";
        else if (ev.type == "End") ph = "E";
        else if (ev.type == "Instant") ph = "i";
        else if (ev.type == "Counter") ph = "C";
        else continue;

        if (ev.class_ == "IdentifyThread")
            continue;

        ChromeArgs args;
        args.event_id = ex.event_id;
        args.parent_id = ex.parent_id;
        args.stream_id = ex.stream_id;
        args.device_id = ex.device_id;
        args.kernel_id = ex.kernel_id;
        args.size = ex.size;
        args.ptr = ex.ptr;
        args.src_ptr = ex.src_ptr;
        args.dst_ptr = ex.dst_ptr;
        args.load_address = ex.load_address;
        args.alignment = ex.alignment;
        args.barrier = ex.barrier;
        args.rsp_type = ex.rsp_type;
        if (ph == "C" && ev.class_ == "MemoryStats") {
            args.allocated = ex.allocated_mem;
            args.free = ex.free_mem;
            args.max_contiguous_free = ex.max_contig_free;
        }

        ChromeEvent ce;
        ce.ph = ph;
        ce.name = ev.class_;
        ce.cat = std::string{category_for(ev.class_)};
        ce.ts = ts;
        ce.pid = 0;
        ce.tid = tid;
        if (ph == "X" && ex.duration_ns)
            ce.dur = *ex.duration_ns / 1000;
        if (ph == "i")
            ce.s = "t";
        ce.args = std::move(args);
        trace.traceEvents.push_back(std::move(ce));

        // -- Synthetic device-side execution span --
        if (ev.class_ == "ResponseReceived"
            && ex.device_cmd_start_ts && ex.device_cmd_exec_dur
            && device_freq > 0)
        {
            int stream = ex.stream_id.value_or(0);
            int64_t start_ticks = static_cast<int64_t>(*ex.device_cmd_start_ts);
            int64_t wait_ticks = ex.device_cmd_wait_dur
                ? static_cast<int64_t>(*ex.device_cmd_wait_dur) : 0;
            int64_t exec_ticks = static_cast<int64_t>(*ex.device_cmd_exec_dur);

            auto dev_ts_opt = dev_to_host_us(start_ticks + wait_ticks);
            if (!dev_ts_opt) continue;
            int64_t dev_ts = *dev_ts_opt;
            int64_t exec_us = exec_ticks * 1000 / device_freq / 1000;

            std::optional<int> kid;
            if (ex.event_id) {
                if (auto it = event_to_kernel.find(*ex.event_id); it != event_to_kernel.end())
                    kid = it->second;
            }

            std::string dev_name;
            if (kid) {
                if (auto it = kernel_names.find(*kid); it != kernel_names.end())
                    dev_name = it->second;
                else
                    dev_name = "Kernel " + std::to_string(*kid);
            } else {
                dev_name = "DeviceExec";
            }

            ChromeArgs dev_args;
            dev_args.event_id = ex.event_id;
            dev_args.kernel_id = kid;
            if (ex.device_cmd_wait_dur)
                dev_args.wait_ns = wait_ticks * 1000 / device_freq;

            ChromeEvent dev_ev;
            dev_ev.ph = "X";
            dev_ev.name = std::move(dev_name);
            dev_ev.cat = "device";
            dev_ev.ts = dev_ts;
            dev_ev.dur = exec_us;
            dev_ev.pid = 1;
            dev_ev.tid = static_cast<uint64_t>(stream);
            dev_ev.args = std::move(dev_args);
            trace.traceEvents.push_back(std::move(dev_ev));
        }
    }

    // -- Write output --
    json out_json = trace;
    auto out = out_json.dump();

    std::cout << "Writing " << out.size() << " bytes to " << output_path.string() << "...\n";
    {
        std::ofstream ofs(output_path, std::ios::binary);
        if (!ofs) {
            std::cerr << "Error: cannot open " << output_path.string() << " for writing\n";
            return 1;
        }
        ofs.write(out.data(), static_cast<std::streamsize>(out.size()));
        if (!ofs) {
            std::cerr << "Error: failed to write to " << output_path.string() << "\n";
            return 1;
        }
    }

    std::cout << "Done.\n";
    return 0;
}
