# et-profile-convert

Convert ET runtime profiler traces to Chrome Trace Event Format for
visualization in `chrome://tracing` or Perfetto.

## Usage

```
et-profile-convert <input.json> [output.json] [-k kernel_id.json]
```

### Arguments

- `input.json` -- ET runtime profiler trace (cereal JSON format).
- `output.json` -- Output path. Defaults to `<input>_chrome.json`.

### Options

- `-k <kernel_id.json>` -- Kernel-name-to-ID mapping file (see below).
  If omitted, the tool looks for `kernel_id.json` in the same directory
  as the input file.
- `-h`, `--help` -- Show help message.

## Kernel ID mapping file

The ET runtime assigns each kernel a numeric ID at load time, but does not
record the kernel name. Without a mapping file, device-side spans in the
trace will show up as "Kernel 0", "Kernel 1", etc.

To get human-readable names, your application must generate a
`kernel_id.json` file that maps kernel names to their numeric IDs. The
format is a flat JSON object:

```json
{
    "my_matmul_kernel": 0,
    "softmax_kernel": 1,
    "relu_kernel": 2
}
```

Keys are kernel name strings, values are the integer IDs that the ET
runtime assigned when loading each kernel. Your application knows both
pieces of information (the name you chose and the ID the runtime returned),
so it is the only component that can produce this mapping.

Place the file next to the input trace for automatic detection, or pass it
explicitly with `-k`.

## Output

The output is a standard Chrome Trace Event Format JSON file with:

- **pid 0 ("Host")** -- Host-side API events (kernel launches, memory
  operations, synchronization, commands) mapped from the ET trace.
- **pid 1 ("Device")** -- Synthetic device-side execution spans
  reconstructed from `ResponseReceived` events. Each span shows the
  actual on-device execution time. Threads correspond to device streams.
- **Metadata events** -- Process and thread names.
- **Counter events** -- Memory statistics (allocated, free, max contiguous
  free) when present in the trace.

### Event categories

Events are grouped into categories for filtering in the viewer:

| Category  | Event classes                                              |
|-----------|------------------------------------------------------------|
| kernel    | KernelLaunch, LoadCode, UnloadCode                         |
| memory    | MallocDevice, FreeDevice, Memcpy*, CmaCopy, MemoryStats    |
| sync      | WaitForEvent, WaitForStream                                |
| command   | CommandSent, ResponseReceived, DispatchEvent, DeviceCommand|
| system    | everything else                                            |

## Clock synchronization

The host and device run independent clocks. The tool computes a linear
mapping (scale + offset) between device ticks and host timestamps using
the lower convex hull of (device_time, host_time) sample pairs from
`ResponseReceived` events. This filters out high-latency outliers and
corrects for both clock offset and drift.

The device clock frequency is read from `GetDeviceProperties` events in
the trace.

## Viewing the output

Open the output JSON in either:

- `chrome://tracing` (paste the URL in a Chromium-based browser)
- [Perfetto UI](https://ui.perfetto.dev) (drag and drop the file)

## Examples

```bash
# Minimal -- auto-names output, auto-detects kernel_id.json
et-profile-convert trace.json

# Explicit output and kernel mapping
et-profile-convert trace.json trace_chrome.json -k my_kernels.json
```
