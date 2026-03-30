# ET Platform Profiling and Metrics Comparison Report

## Overview

This report compares the data available from two profiling/metrics sources in the ET Platform:

1. **Runtime Profiler Traces** - Host-Side event-driven profiling via the Esperanto runtime library
2. **`et-metrics` Device Polling** - Device-Side hardware metrics collected via the Service Processor

---

## Runtime Profiler Traces

### Source
- **Repository:** `et-platform/esperanto-tools-libs`
- **Implementation:** `src/RuntimeImp.cpp`, `src/MemoryManager.cpp`
- **Output:** Cereal JSON format (`et_runtime_trace.json` when using `GGML_ET_PROFILE` env var)
- **Reference:** [`ggml-et.cpp:171-197`](https://github.com/aifoundry-org/et-platform/blob/master/esperanto-tools-libs/tools/src/et_profile_convert.cpp)

### Data Captured

| Metric | Description | Accuracy |
|--------|-------------|----------|
| **Memory Stats** | `allocated_memory`, `free_memory`, `max_contiguous_free_mem` | ✅ Accurate |
| **Kernel Launches** | Kernel ID, load address, stream, device, timestamp | ✅ Accurate |
| **Memcpy Operations** | H2D, D2H, D2D transfers with sizes, addresses, timestamps | ✅ Accurate |
| **DMA Operations** | Command send/receive times, device command wait/exec durations | ✅ Accurate |
| **Stream/Event Operations** | Create, destroy, wait for stream/event | ✅ Accurate |
| **Device Properties** | Full device specs (frequency, memory size, cache sizes, etc.) | ✅ Accurate |
| **Thread Identification** | Thread names and IDs | ✅ Accurate |
| **Response Types** | DMARead, DMAWrite, Kernel, DMAP2P | ✅ Accurate |

### Memory Stats Implementation

Memory statistics are calculated in `MemoryManager.cpp`:

```cpp
// File: esperanto-tools-libs/src/MemoryManager.cpp:138-149
size_t MemoryManager::getFreeBytes() const {
  return std::accumulate(begin(free_), end(free_), 0UL,
                         [](const auto& accumulate, const auto& e) { return accumulate + e.size_; }) *
         getBlockSize();
}

size_t MemoryManager::getAllocatedBytes() const {
  return std::accumulate(begin(allocated_), end(allocated_), 0UL,
                         [](const auto& accumulate, const auto& e) { return accumulate + e.second; }) *
         getBlockSize();
}
```

**Reference:** [`MemoryManager.cpp`](https://github.com/aifoundry-org/et-platform/blob/master/esperanto-tools-libs/src/MemoryManager.cpp)

### Characteristics

| Feature | Status |
|---------|--------|
| Event-driven | ✅ Yes |
| Per-operation timestamps | ✅ Yes |
| Accurate memory tracking | ✅ Yes |
| Hardware sensors (temp/voltage/power) | ❌ No |
| Resource utilization metrics | ❌ No |
| Device-side execution details | ❌ No |

---

## `et-metrics` Device Polling

Experimental branch based on `et-top` except it logs data to .csv file.

### Source
- **Repository:** `et-platform/device-management-application`
- **Implementation:** `src/et-metrics.cc`
- **Data Source:** Device Management API via Service Processor
- **Reference:** [`et-metrics.cc`](https://github.com/ubergarm/et-platform/tree/ug/vibe-et-metrics/device-management-application/src/et-metrics.cc)

### Data Captured

| Metric | Description | Source Function |
|--------|-------------|-----------------|
| **CMA Allocated** | **Host RAM** allocated for DMA buffers (MB) - *Not device LPDDR4* | Device layer attribute file |
| **CMA Allocation Rate** | Host RAM allocation rate for DMA buffers (MB/sec) | Device layer attribute file |
| **SP Stats** | Power/temp/voltage/freq for: System, Minion, SRAM, NOC | `get_sp_stats()` via DM API |
| **MM Stats** | Compute resource utilization & bandwidth | `get_mm_stats()` via DM API |
| **ASIC Frequencies** | Minion, NOC, MemShire, DDR, PCIe, IO frequencies | `get_asic_frequencies()` |
| **Module/ASIC Voltage** | Voltage readings for all domains | `get_module_voltage()`, `get_asic_voltage()` |
| **VQ Stats** | Submission Queue message counts, rates, utilization | Device layer attribute files |
| **Device Details** | Card ID, firmware versions | Asset info & FW version commands |

### Characteristics

| Feature | Status |
|---------|--------|
| Hardware sensors | ✅ Yes |
| Resource utilization | ✅ Yes |
| Bandwidth metrics | ✅ Yes |
| Frequency monitoring | ✅ Yes |
| Polling-based | ✅ Yes (default 1000ms intervals) |
| Kernel-level detail | ❌ No |
| Memory fragmentation info | ❌ No |

### Important: CMA Memory is Host-Side

**CMA (Contiguous Memory Allocator)** is a **Linux kernel feature** that reserves physically contiguous memory regions on the **host system (x86_64)** for DMA operations.

| Memory Type | Location | Purpose |
|-------------|----------|---------|
| **CMA Memory** | Host RAM (x86_64) | DMA buffers for PCIe transfers between host and device |
| **LPDDR4 (DRAM)** | ET-SoC-1 Device | Working memory for Minion/CM kernels running on the device |

The `cma_allocated` metric tracks **how much host RAM** has been allocated for DMA buffers, **NOT** how much device LPDDR4 is in use.

**Implementation Reference:**
- [`DevicePcie.cpp:51-67`](https://github.com/aifoundry-org/et-platform/blob/master/devicelayer/src/DevicePcie.cpp#L51-L67) - Reads `/proc/meminfo` for host CMA pool
- [`et_sysfs_mem_stats.c`](https://github.com/aifoundry-org/et-platform/blob/master/et-driver/et_sysfs_mem_stats.c) - Kernel driver tracks CMA allocations

---

## Incomplete/Hardcoded Metrics

### Critical Issues

The following functions in `device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c` contain **hardcoded values** instead of actual implementations:

### 1. DRAM Capacity Percentage (Hardcoded to 80%)

```c
// File: device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c:349-352
int update_dram_capacity_percent(void)
{
    // TODO : Compute DRAM capacity utilization. Update the global.
    // https://esperantotech.atlassian.net/browse/SW-6608
    get_soc_perf_reg()->dram_capacity_percent = 80;  // HARDCODED!
    return 0;
}
```

**Git Reference:** [`perf_mgmt.c:349-352`](https://github.com/aifoundry-org/et-platform/blob/master/device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c#L349-L352)

### 2. DRAM Bandwidth (Hardcoded to 16 req/sec)

```c
// File: device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c:70-88
int update_dram_bw(void)
{
    // FIXME: Upon SW-5688 resolution delete below two lines and uncomment lines above
    get_soc_perf_reg()->dram_bw.read_req_sec = 16;   // HARDCODED!
    get_soc_perf_reg()->dram_bw.write_req_sec = 16;  // HARDCODED!
    return 0;
}
```

**Git Reference:** [`perf_mgmt.c:70-88`](https://github.com/aifoundry-org/et-platform/blob/master/device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c#L70-L88)

### 3. ASIC Per-Core Utilization (Stub - Not Implemented)

```c
// File: device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c:402-409
int get_asic_per_core_util(uint8_t *core_util)
{
    // TODO : Finalize the payload and implement the function get_asic_per_core_util()
    // to return payload.
    // https://esperantotech.atlassian.net/browse/SW-6608
    (void)core_util;
    return 0;
}
```

**Git Reference:** [`perf_mgmt.c:402-409`](https://github.com/aifoundry-org/et-platform/blob/master/device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c#L402-L409)

### 4. ASIC Utilization (Stub - Not Implemented)

```c
// File: device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c:430-437
int get_asic_utilization(uint8_t *asic_util)
{
    // TODO : Finalize the payload and implement the function get_asic_utilization()
    // to return payload
    // https://esperantotech.atlassian.net/browse/SW-6608
    (void)asic_util;
    return 0;
}
```

**Git Reference:** [`perf_mgmt.c:430-437`](https://github.com/aifoundry-org/et-platform/blob/master/device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c#L430-L437)

### 5. ASIC Stalls (Stub - Not Implemented)

```c
// File: device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c:446-453
int get_asic_stalls(uint8_t *asic_stalls)
{
    // TODO : Finalize the payload and implement the function get_asic_stalls()
    // to return payload
    // https://esperantotech.atlassian.net/browse/SW-6608
    (void)asic_stalls;
    return 0;
}
```

**Git Reference:** [`perf_mgmt.c:446-453`](https://github.com/aifoundry-org/et-platform/blob/master/device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c#L446-L453)

### 6. ASIC Latency (Stub - Not Implemented)

```c
// File: device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c:464-471
int get_asic_latency(uint8_t *asic_latency)
{
    // TODO : Finalize the payload and implement the function get_asic_latency()
    // to return payload
    // https://esperantotech.atlassian.net/browse/SW-6608
    (void)asic_latency;
    return 0;
}
```

**Git Reference:** [`perf_mgmt.c:464-471`](https://github.com/aifoundry-org/et-platform/blob/master/device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c#L464-L471)

---

## Comparison Summary

| Feature | Runtime Profiler | et-metrics |
|---------|-----------------|------------|
| Device DRAM allocated/free | ✅ Accurate | ❌ No (see hardcoded DRAM capacity below) |
| Host CMA memory | ❌ No | ⚠️ Tracks host RAM for DMA buffers only |
| Memory fragmentation | ✅ Max contiguous free | ❌ No |
| Kernel launches | ✅ Detailed | ❌ No |
| Memcpy operations | ✅ Detailed | ❌ No |
| Temperature | ❌ No | ✅ Yes |
| Voltage | ❌ No | ✅ Yes |
| Power | ❌ No | ✅ Yes |
| CM utilization | ❌ No | ✅ Yes (avg/min/max) |
| Bandwidth (DDR/PCIe/L2L3) | ❌ No | ✅ Yes (avg/min/max) |
| Frequencies | ❌ No | ✅ Yes (all domains) |
| DRAM capacity % | ❌ No | ⚠️ **Hardcoded (80%)** |
| DRAM bandwidth | ❌ No | ⚠️ **Hardcoded (16 req/sec)** |
| Per-core utilization | ❌ No | ⚠️ **Stub (not implemented)** |
| Event-driven | ✅ Yes | ❌ Polling (1s intervals) |

---

## Recommendations

1. **Use Both Sources Together**
   - Runtime profiler for application-level performance analysis (kernel timing, memory usage, data transfers)
   - `et-metrics` for hardware health monitoring (thermal, power, utilization, frequencies)

2. **Incomplete Metrics Need Implementation**
   - The hardcoded values in `perf_mgmt.c` should be replaced with actual sensor readings
   - Jira ticket SW-6608 tracks some of these incomplete implementations
   - DRAM capacity and bandwidth metrics are currently unreliable for production use

3. **For Accurate Memory Monitoring**
   - Use the runtime profiler's `MemoryStats` events for accurate memory tracking
   - The `et-metrics` CMA allocated value is less detailed (no fragmentation info)

---

## File References

| Component | File Path |
|-----------|-----------|
| Runtime Profiler Implementation | `esperanto-tools-libs/src/RuntimeImp.cpp` |
| Memory Manager | `esperanto-tools-libs/src/MemoryManager.cpp` |
| Profiler Interface | `esperanto-tools-libs/include/runtime/IProfiler.h` |
| et-metrics Implementation | `device-management-application/src/et-metrics.cc` |
| Performance Management (Hardcoded) | `device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c` |
| Device Management API Types | `device-api/include/management-api/device_mgmt_api_rpc_types.h` |
| Trace Layout | `et-trace/include/et-trace/layout.h` |

---

*Report generated from analysis of the et-platform repository.*
