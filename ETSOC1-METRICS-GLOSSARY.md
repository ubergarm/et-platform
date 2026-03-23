# ET-SoC-1 Metrics Glossary
*NOTE*: This document was vibe coded using [ubergarm/Qwen3.5-122B-A10B-GGUF IQ5_KS 77.341 GiB (5.441 BPW)](https://huggingface.co/ubergarm/Qwen3.5-122B-A10B-GGUF#iq5_ks-77341-gib-5441-bpw) with `opencode`.

## Overview

Metrics are collected from two primary data structures:

1. **`op_stats_t`** - Operating Point Statistics (Power, Temperature, Voltage, Frequency)
2. **`compute_resources_sample`** - Compute Resource Statistics (Utilization, Bandwidth)

### Sampling Configuration

| Parameter | Value | Source |
|-----------|-------|--------|
| **SP Stats Sampling Rate** | 10ms (100 Hz) | `DM_TASK_DELAY_MS` in `mgmt_build_config.h` |
| **MM Stats Sampling Rate** | ~1ms (1000 Hz) | `STATW_Launch` periodic worker |
| **Moving Average Window** | 100 samples | `STATW_BW_CMA_SAMPLE_COUNT` |
| **Cache Line Size** | 64 bytes | `CACHE_LINE_SIZE` |
| **DDR Frequency** | 933 MHz | `DDR_FREQUENCY` constant |

---

## op_stats_t - Operating Point Statistics

**Source:** Service Processor firmware (`thermal_pwr_mgmt.c`)  
**Collection:** `DM_CMD_GET_SP_STATS`  
**Trace Type:** `TRACE_CUSTOM_TYPE_SP_OP_STATS`

### Structure Layout

```c
struct op_stats_t {
    struct op_module minion;   // Minion Shire stats
    struct op_module sram;     // L2 Cache (SRAM) stats  
    struct op_module noc;      // Network-on-Chip stats
    struct op_module system;   // System-level stats
};

struct op_module {
    struct op_value temperature;  // °C
    struct op_value power;        // See units below
    struct op_value voltage;      // See units below
    struct op_value freq;         // MHz
};

struct op_value {
    uint16_t avg;  // Moving average
    uint16_t min;  // Minimum observed
    uint16_t max;  // Maximum observed
};
```

### Metrics

#### System Module (`op_stats_t.system`)

| Metric | Unit | Raw Unit | Conversion | Description |
|--------|------|----------|------------|-------------|
| `system.temperature` | °C | °C | None | System-wide temperature (from PVT sensors) |
| `system.power` | Watts | 10mW | `power / 100.0` | Total system power consumption |
| `system.frequency` | MHz | MHz | None | System reference frequency |

**Hardware Reference:** Section 1.6 "Process, Voltage, and Temperature Monitoring" - ET-SoC-1 contains 36 temperature sensors (1 per Minion/IO Shire) interfacing to 5 PVT controllers in the I/O Shire.

#### Minion Module (`op_stats_t.minion`)

| Metric | Unit | Raw Unit | Conversion | Description |
|--------|------|----------|------------|-------------|
| `minion.temperature` | °C | °C | None | Minion Shire temperature |
| `minion.power` | Watts | mW | `power / 1000.0` | Minion core power consumption |
| `minion.voltage` | mV | Binary | `250 + (value × 5) / 1` | Minion supply voltage (PMIC reading) |
| `minion.frequency` | MHz | MHz | None | Minion Shire clock frequency |

**Voltage Formula:** `BIN2VOLTAGE(REG_VALUE, 250, 5, 1)` = `250 + (REG_VALUE × 5)` mV  
**PMIC Reference:** `PMIC_MINION_VOLTAGE_BASE=250`, `PMIC_MINION_VOLTAGE_MULTIPLIER=5`

#### SRAM Module (`op_stats_t.sram`)

| Metric | Unit | Raw Unit | Conversion | Description |
|--------|------|----------|------------|-------------|
| `sram.temperature` | °C | °C | None | L2 cache temperature (not hardware-supported) |
| `sram.power` | Watts | mW | `power / 1000.0` | L2 SRAM power consumption |
| `sram.voltage` | mV | Binary | `250 + (value × 5) / 1` | SRAM supply voltage |
| `sram.frequency` | MHz | MHz | None | L2 cache clock frequency |

**Note:** SRAM temperature is marked as "not supported by hardware" in trace code comments.

#### NOC Module (`op_stats_t.noc`)

| Metric | Unit | Raw Unit | Conversion | Description |
|--------|------|----------|------------|-------------|
| `noc.temperature` | °C | °C | None | Network-on-Chip temperature (not hardware-supported) |
| `noc.power` | Watts | mW | `power / 1000.0` | NOC power consumption |
| `noc.voltage` | mV | Binary | `250 + (value × 5) / 1` | NOC supply voltage |
| `noc.frequency` | MHz | MHz | None | NOC clock frequency |

**Note:** NOC temperature is marked as "not supported by hardware" in trace code comments.

---

## compute_resources_sample - Compute Resource Statistics

**Source:** Master Minion firmware (`statw.c` - Stats Worker)  
**Collection:** `DM_CMD_GET_MM_STATS`  
**Trace Type:** `TRACE_CUSTOM_TYPE_MM_COMPUTE_RESOURCES`

### Structure Layout

```c
struct compute_resources_sample {
    struct resource_value cm_utilization;        // Compute Minion utilization (%)
    struct resource_value cm_bw;                 // Compute Minion bandwidth (MB/s)
    struct resource_value pcie_dma_read_bw;      // PCIe DMA read bandwidth (MB/s)
    struct resource_value pcie_dma_read_utilization;  // PCIe DMA read util (%)
    struct resource_value pcie_dma_write_bw;     // PCIe DMA write bandwidth (MB/s)
    struct resource_value pcie_dma_write_utilization; // PCIe DMA write util (%)
    struct resource_value ddr_read_bw;           // DDR read bandwidth (MB/s)
    struct resource_value ddr_write_bw;          // DDR write bandwidth (MB/s)
    struct resource_value l2_l3_read_bw;         // L2/L3 read bandwidth (MB/s)
    struct resource_value l2_l3_write_bw;        // L2/L3 write bandwidth (MB/s)
};

struct resource_value {
    uint64_t avg;  // Moving average (100-sample CMA)
    uint64_t min;  // Minimum observed
    uint64_t max;  // Maximum observed
};
```

### Metrics

#### Compute Resources

| Metric | Unit | Formula | Description |
|--------|------|---------|-------------|
| `cm_utilization` | % (0-100) | `accumulated_cycles × 100 / sampling_interval_cycles` | Compute Minion utilization percentage. Uses `STATW_PERCENTAGE_MULTIPLIER = 100` |
| `cm_bw` | MB/s | See bandwidth formula | Aggregate Compute Minion bandwidth |

#### PCIe DMA Resources

| Metric | Unit | Formula | Description |
|--------|------|---------|-------------|
| `pcie_dma_read_bw` | MB/s | `(req_count × 64 × freq_mhz) / cycles` | PCIe DMA read bandwidth from host perspective |
| `pcie_dma_write_bw` | MB/s | `(req_count × 64 × freq_mhz) / cycles` | PCIe DMA write bandwidth from host perspective |
| `pcie_dma_read_utilization` | % (0-100) | N/A | PCIe DMA read channel utilization |
| `pcie_dma_write_utilization` | % (0-100) | N/A | PCIe DMA write channel utilization |

**Note:** "DMA read is DMA write and DMA write is DMA read from host's perspective" (statw.c comment)

#### DDR Memory Resources

| Metric | Unit | Formula | Description |
|--------|------|---------|-------------|
| `ddr_read_bw` | MB/s | `(req_count × 64 × 933) / cycles` | DDR memory read bandwidth |
| `ddr_write_bw` | MB/s | `(req_count × 64 × 933) / cycles` | DDR memory write bandwidth |

**DDR Frequency:** Fixed at 933 MHz (from `DDR_FREQUENCY` constant)  
**Calculation:** Based on PMU counters from Memory Shire (8 DDR controllers)

#### L2/L3 Cache Resources

| Metric | Unit | Formula | Description |
|--------|------|---------|-------------|
| `l2_l3_read_bw` | MB/s | `(req_count × 64 × minion_freq) / cycles` | L2/L3 cache read bandwidth |
| `l2_l3_write_bw` | MB/s | `(req_count × 64 × minion_freq) / cycles` | L2/L3 cache write bandwidth |

**Architecture:** 34 Minion Shires, each with 4 MB L2 cache (configurable as L2/L3)

---

## Bandwidth Calculation Formula

All bandwidth metrics use the same formula from `STATW_PMU_REQ_COUNT_TO_MBPS`:

```
bandwidth_MB_s = (request_count × CACHE_LINE_SIZE × frequency_MHz) / cycles_consumed

Where:
  - request_count: PMU counter value (64-byte requests)
  - CACHE_LINE_SIZE: 64 bytes
  - frequency_MHz: Counter reference frequency (DDR=933, Minion=variable)
  - cycles_consumed: Sampling interval in cycles
```

**Example for DDR:**
```
ddr_read_bw = (pmc0 × 64 × 933) / cycle_count  [MB/s]
```

---

## Moving Average Calculation

All metrics use a **Commutative Moving Average (CMA)** with 100 samples:

```c
cma = (current_value + old_value × (100 - 1)) / 100
```

**Source:** `statw_recalculate_cma()` in `statw.c`  
**Sample Count:** `STATW_BW_CMA_SAMPLE_COUNT = 100`

### Implementation Details

The CMA is calculated **on-device by the Master Minion Stats Worker** (`statw.c:189-204`), not on the host:

```c
static inline uint64_t statw_recalculate_cma(
    uint64_t old_value, uint64_t current_value, uint64_t sample_count)
{
    if (current_value > old_value)
    {
        double cma_temp = ceil(((double)current_value + (double)(old_value * (sample_count - 1))) /
                               (double)sample_count);
        return (uint64_t)cma_temp;
    }
    else if (current_value < old_value)
    {
        double cma_temp = floor(((double)current_value + (double)(old_value * (sample_count - 1))) /
                                (double)sample_count);
        return (uint64_t)cma_temp;
    }
    return old_value;
}
```

**Key Points:**

1. **Firmware-side calculation:** The CMA is computed in the `STATW_Launch` worker loop (~1ms interval)
2. **Per-sample updates:** Each sampling iteration calls `STATW_RECALC_CMA_MIN_MAX` macro which invokes `statw_recalculate_cma()`
3. **Applied to bandwidth metrics:** `ddr_read_bw`, `ddr_write_bw`, `l2_l3_read_bw`, `l2_l3_write_bw` (via `statw_update_cma()`)
4. **Applied to utilization metrics:** `cm_utilization`, `pcie_dma_read_utilization`, `pcie_dma_write_utilization` (via `statw_fill_utilization_stats()`)
5. **Host receives pre-calculated values:** The host (`et-metrics.cc`, `et-top.cc`) only retrieves the already-smoothed `avg` values via `DM_CMD_GET_MM_STATS` - no recalculation occurs on the host side

**Call Sites in `statw.c`:**

- Line 62: Macro `STATW_RECALC_CMA_MIN_MAX` used for bandwidth metrics
- Line 189: `statw_recalculate_cma()` function definition
- Line 311-329: `statw_update_cma()` applies CMA to DDR and L2/L3 bandwidth
- Line 680: `STATW_Add_New_Sample_Atomically()` applies CMA to PCIe DMA bandwidth
- Line 720-735: `statw_fill_utilization_stats()` applies CMA to utilization metrics

---

## Frequency Metrics (Separate Command)

**Command:** `DM_CMD_GET_ASIC_FREQUENCIES`

| Metric | Unit | Description |
|--------|------|-------------|
| `minion_shire_mhz` | MHz | Minion Shire clock frequency |
| `noc_mhz` | MHz | Network-on-Chip frequency |
| `mem_shire_mhz` | MHz | Memory Shire frequency |
| `ddr_mhz` | MHz | DDR memory frequency |
| `pcie_shire_mhz` | MHz | PCIe Shire frequency |
| `io_shire_mhz` | MHz | IO Shire (Maxion) frequency |

**Source:** PLL frequencies queried via `get_pll_frequency()`

---

## Voltage Metrics (Separate Commands)

### Module Voltage (`DM_CMD_GET_MODULE_VOLTAGE`)

Raw binary values from PMIC. Conversion depends on rail:

| Rail | Base (mV) | Multiplier | Divider | Formula |
|------|-----------|------------|---------|---------|
| DDR | 250 | 5 | 1 | `250 + (value × 5)` |
| SRAM/L2 | 250 | 5 | 1 | `250 + (value × 5)` |
| Minion | 250 | 5 | 1 | `250 + (value × 5)` |
| NOC | 250 | 5 | 1 | `250 + (value × 5)` |
| VDDQLP | 250 | 10 | 1 | `250 + (value × 10)` |
| VDDQ | 250 | 10 | 1 | `250 + (value × 10)` |
| PCIe Logic | 600 | 625 | 100 | `600 + (value × 6.25)` |

### ASIC Voltage (`DM_CMD_GET_ASIC_VOLTAGE`)

Similar binary encoding, includes additional rails (Maxion, PShire, IOShire).

---

## Summary Table

| Structure | Command | Sampling Rate | Metrics | Units |
|-----------|---------|---------------|---------|-------|
| `op_stats_t` | `DM_CMD_GET_SP_STATS` | 10ms (100 Hz) | Power, Temp, Voltage, Freq | W, °C, mV, MHz |
| `compute_resources_sample` | `DM_CMD_GET_MM_STATS` | ~1ms (1000 Hz) | Utilization, Bandwidth | %, MB/s |
| `asic_frequencies_t` | `DM_CMD_GET_ASIC_FREQUENCIES` | On-demand | Clock frequencies | MHz |
| `module_voltage_t` | `DM_CMD_GET_MODULE_VOLTAGE` | On-demand | PMIC voltages | Binary → mV |
| `asic_voltage_t` | `DM_CMD_GET_ASIC_VOLTAGE` | On-demand | ASIC voltages | Binary → mV |

---

## References

### Source Files

| File | Purpose |
|------|---------|
| `et-platform/et-trace/include/et-trace/layout.h` | Structure definitions (`op_stats_t`, `compute_resources_sample`, `trace_custom_event_t`) |
| `et-platform/device-api/include/management-api/device_mgmt_api_spec.h` | Device Management API command definitions |
| `et-platform/device-api/include/management-api/device_mgmt_api_rpc_types.h` | RPC type definitions for all metric structures |
| `et-platform/device-bootloaders/src/ServiceProcessorBL2/rtos_task/dm_task.c` | SP stats collection task (10ms interval) |
| `et-platform/device-bootloaders/src/ServiceProcessorBL2/services/thermal_pwr_mgmt.c` | Power/temperature/voltage measurement implementation |
| `et-platform/device-bootloaders/src/ServiceProcessorBL2/services/perf_mgmt.c` | Performance management (MM stats retrieval) |
| `et-platform/device-bootloaders/src/ServiceProcessorBL2/include/bl2_pmic_controller.h` | PMIC voltage conversion constants |
| `et-platform/device-minion-runtime/src/MasterMinion/src/workers/statw.c` | MM stats worker (bandwidth/utilization sampling) |
| `et-platform/device-minion-runtime/src/MasterMinion/include/workers/statw.h` | MM stats configuration constants |
| `et-platform/device-management-application/src/et-metrics.cc` | Host-side metrics collection and CSV output |
| `et-platform/device-management-application/include/et-metrics.hpp` | Metrics collector class definition |

### Documentation

| Document | Section |
|----------|---------|
| [ET Programmer's Reference Manual PDF](https://github.com/aifoundry-org/et-man/blob/main/ET%20Programmer's%20Reference%20Manual.pdf) | Section 1.6: Process, Voltage, and Temperature Monitoring |
| `trace-utils/README.md` | Trace tools overview |
| `trace-utils/src/et_trace/txt_trace_sink.cc` | Trace-to-text conversion (metric formatting) |
| `trace-utils/src/et_trace/perfetto_trace_sink.cc` | Trace-to-Perfetto conversion |

### Key Constants

```c
// Sampling intervals
#define DM_TASK_DELAY_MS 10                    // 10ms SP stats interval
#define STATW_BW_CMA_SAMPLE_COUNT 100UL        // 100-sample moving average
#define STATW_PERCENTAGE_MULTIPLIER 100        // Utilization percentage multiplier
#define DDR_FREQUENCY 933UL                    // DDR frequency in MHz

// Power conversion
#define k10mWToWDivisor 1000                   // System power: 10mW → W
#define kmWToWDivisor 1000                     // Module power: mW → W

// Voltage conversion (PMIC)
#define PMIC_MINION_VOLTAGE_BASE 250
#define PMIC_MINION_VOLTAGE_MULTIPLIER 5
#define PMIC_GENERIC_VOLTAGE_DIVIDER 1

// Hardware architecture
#define CACHE_LINE_SIZE 64                     // Bytes
// NUM_SHIRES = 34 (32 compute + 1 master + 1 spare)
// NUM_MEM_SHIRES = 8
// BANKS_PER_SC = 4
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ET-SoC-1 Device                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────┐         ┌──────────────────────┐              │
│  │   Service Processor  │         │   Master Minion      │              │
│  │   (Firmware)         │         │   (Firmware)         │              │
│  │                      │         │                      │              │
│  │  ┌──────────────┐    │         │  ┌──────────────┐    │              │
│  │  │ dm_task      │    │         │  │ statw.c      │    │              │
│  │  │ (10ms loop)  │    │         │  │ (~1ms loop)  │    │              │
│  │  └──────┬───────┘    │         │  └──────┬───────┘    │              │
│  │         │            │         │         │            │              │
│  │  ┌──────▼───────┐    │         │  ┌──────▼───────┐    │              │
│  │  │ thermal_pwr  │    │         │  │ PMU Counters │    │              │
│  │  │ _mgmt.c      │    │         │  │ (DDR, L2/L3) │    │              │
│  │  └──────┬───────┘    │         │  └──────┬───────┘    │              │
│  │         │            │         │         │            │              │
│  │  ┌──────▼───────┐    │         │  ┌──────▼───────┐    │              │
│  │  │ op_stats_t   │    │         │  │ compute_     │    │              │
│  │  └──────┬───────┘    │         │  │ resources_   │    │              │
│  │         │            │         │  │ sample       │    │              │
│  │         ▼            │         │  └──────┬───────┘    │              │
│  │  ┌──────────────┐    │         │         │            │              │
│  │  │ Trace Buffer │◄───┼─────────┼─────────┤            │              │
│  │  │ (SP Stats)   │    │         │         │            │              │
│  │  └──────────────┘    │         │  ┌──────▼───────┐    │              │
│  │                      │         │  │ Trace Buffer │    │              │
│  │                      │         │  │ (MM Stats)   │    │              │
│  │                      │         │  └──────────────┘    │              │
│  └──────────┬───────────┘         └───────────┬──────────┘              │
│             │                                 │                         │
└─────────────┼─────────────────────────────────┼─────────────────────────┘
              │                                 │
              │  DeviceManagement API           │
              │  (serviceRequest)               │
              ▼                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Host (Linux)                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      et-metrics executable                       │   │
│  │                                                                  │   │
│  │  DM_CMD_GET_SP_STATS ──► op_stats_t ──► CSV output               │   │
│  │  DM_CMD_GET_MM_STATS ──► compute_resources_sample ──► CSV        │   │
│  │  DM_CMD_GET_ASIC_FREQUENCIES ──► asic_frequencies_t ──► CSV      │   │
│  │  DM_CMD_GET_MODULE_VOLTAGE ──► module_voltage_t ──► CSV          │   │
│  │  DM_CMD_GET_ASIC_VOLTAGE ──► asic_voltage_t ──► CSV              │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Related Tools

| Tool | Purpose |
|------|---------|
| [et-metrics](https://github.com/ubergarm/et-platform/tree/ug/vibe-et-metrics) | Real-time metrics collection to CSV |
| `et-top` | Interactive terminal UI for device metrics |
| `trace-utils/extract_metrics.py` | Extract utilization metrics from runtime traces |
| `trace-utils/dt2json` | Convert device traces to JSON/Perfetto |
| `trace-utils/evt2json` | Convert event traces to JSON/Perfetto |
