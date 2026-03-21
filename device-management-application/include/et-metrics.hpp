/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------*/
#pragma once
#include "device-layer/IDeviceLayer.h"
#include "deviceManagement/DeviceManagement.h"
#include "esperanto/et-trace/layout.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <esperanto/et-trace/decoder.h>
#include <fstream>
#include <functional>
#include <hostUtils/logging/Logging.h>
#include <glog/logging.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <atomic>

#define DV_LOG(severity) ET_LOG(ET_METRICS, severity)
#define DV_DLOG(severity) ET_DLOG(ET_METRICS, severity)
#define DV_VLOG(level) ET_VLOG(ET_METRICS, level)
#define ET_METRICS "et-metrics"
#define BIN2VOLTAGE(REG_VALUE, BASE, MULTIPLIER, DIVIDER) (BASE + ((REG_VALUE * MULTIPLIER) / DIVIDER))
#define POWER_10MW_TO_W(pwr10mw) (pwr10mw / 100.0)
#define POWER_MW_TO_W(pwrMw) (pwrMw / 1000.0)

static const uint32_t kDmServiceRequestTimeout = 100000;
static const uint32_t kUpdateDelayMS = 1000;
static const int32_t kMaxDeviceNum = 63;

struct sp_stats_t {
  op_stats_t op;
};

struct mm_stats_t {
  uint64_t cycle;
  compute_resources_sample computeResources;
};

struct mem_stats_t {
  uint64_t cmaAllocated;
  uint64_t cmaAllocationRate;
};

struct vq_stats_t {
  std::string qname;
  uint64_t msgCount;
  uint64_t msgRate;
  uint64_t utilPercent;
};

class EtMetrics {
public:
  EtMetrics(int devNum, std::unique_ptr<dev::IDeviceLayer>& dl, device_management::DeviceManagement& dm,
            std::ostream& output, bool outputHeaderOnce, bool noHeader);
  void collectStats(void);
  void outputStats(void);
  bool stopStats(void);
  void requestStop(void);

private:
  void collectDeviceDetails(void);
  void collectMemStats(void);
  void collectVqStats(void);
  void collectSpStats(void);
  void collectMmStats(void);
  void collectFreqStats(void);
  void collectVoltStats(void);
  void outputCsvHeader(void);
  void outputCsvRow(void);
  std::string formatVersion(uint32_t ver);

  int devNum_;
  std::ostream& output_;
  bool outputHeaderOnce_;
  bool noHeader_;
  bool stop_;
  bool refreshDeviceDetails_;
  bool headerOutput_;
  std::string cardId_;
  std::string fwVersion_;
  std::string pmicFwVersion_;
  std::unique_ptr<dev::IDeviceLayer>& dl_;
  device_management::DeviceManagement& dm_;

  std::array<vq_stats_t, 3> vqStats_;
  struct mem_stats_t memStats_;
  struct sp_stats_t spStats_;
  struct mm_stats_t mmStats_;
  device_mgmt_api::asic_frequencies_t freqStats_;
  device_mgmt_api::module_voltage_t moduleVoltStats_;
  device_mgmt_api::asic_voltage_t asicVoltStats_;
};