/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------*/
#define ET_METRICS_IMPL
#include "et-metrics.hpp"
#include <algorithm>
#include <csignal>
#include <iomanip>
#include <regex>

static std::atomic<bool> g_stopRequested(false);

static void signalHandler(int signum) {
  g_stopRequested.store(true);
}

EtMetrics::EtMetrics(int devNum, std::unique_ptr<dev::IDeviceLayer>& dl,
                     device_management::DeviceManagement& dm, std::ostream& output,
                     bool outputHeaderOnce, bool noHeader)
  : devNum_(devNum)
  , output_(output)
  , outputHeaderOnce_(outputHeaderOnce)
  , noHeader_(noHeader)
  , stop_(false)
  , refreshDeviceDetails_(true)
  , headerOutput_(false)
  , dl_(dl)
  , dm_(dm) {
  vqStats_[0].qname = "SQ0:";
  vqStats_[1].qname = "SQ1:";
  vqStats_[2].qname = "CQ0:";
  mmStats_.cycle = 0;
}

bool EtMetrics::stopStats(void) {
  return stop_ || g_stopRequested.load();
}

void EtMetrics::requestStop(void) {
  stop_ = true;
}

void EtMetrics::collectStats(void) {
  if (refreshDeviceDetails_) {
    collectDeviceDetails();
    refreshDeviceDetails_ = false;
  }
  collectMemStats();
  collectVqStats();
  collectSpStats();
  collectMmStats();
  collectFreqStats();
  collectVoltStats();
}

void EtMetrics::collectMemStats(void) {
  std::string dummy;
  std::istringstream attrFileAlloc(dl_->getDeviceAttribute(devNum_, "mem_stats/cma_allocated"));
  for (std::string line; std::getline(attrFileAlloc, line);) {
    std::stringstream ss;
    ss << line;
    ss >> memStats_.cmaAllocated >> dummy;
  }

  std::istringstream attrFileRate(dl_->getDeviceAttribute(devNum_, "mem_stats/cma_allocation_rate"));
  for (std::string line; std::getline(attrFileRate, line);) {
    std::stringstream ss;
    ss << line;
    ss >> memStats_.cmaAllocationRate >> dummy;
  }
}

void EtMetrics::collectVqStats(void) {
  std::string qname;
  uint64_t num;
  std::string dummy;

  std::istringstream attrFileCount(dl_->getDeviceAttribute(devNum_, "ops_vq_stats/msg_count"));
  for (std::string line; std::getline(attrFileCount, line);) {
    std::stringstream ss;
    ss << line;
    ss >> qname >> num >> dummy;
    auto it = std::find_if(vqStats_.begin(), vqStats_.end(), [qname](const auto& e) { return qname == e.qname; });
    if (it != vqStats_.end()) {
      (*it).msgCount = num;
    }
  }

  std::istringstream attrFileRate(dl_->getDeviceAttribute(devNum_, "ops_vq_stats/msg_rate"));
  for (std::string line; std::getline(attrFileRate, line);) {
    std::stringstream ss;
    ss << line;
    ss >> qname >> num >> dummy;
    auto it = std::find_if(vqStats_.begin(), vqStats_.end(), [qname](const auto& e) { return qname == e.qname; });
    if (it != vqStats_.end()) {
      (*it).msgRate = num;
    }
  }

  std::istringstream attrFileUtil(dl_->getDeviceAttribute(devNum_, "ops_vq_stats/utilization_percent"));
  for (std::string line; std::getline(attrFileUtil, line);) {
    std::stringstream ss;
    ss << line;
    ss >> qname >> num >> dummy;
    auto it = std::find_if(vqStats_.begin(), vqStats_.end(), [qname](const auto& e) { return qname == e.qname; });
    if (it != vqStats_.end()) {
      (*it).utilPercent = num;
    }
  }
}

void EtMetrics::collectSpStats(void) {
  uint32_t hostLatency;
  uint64_t deviceLatency;
  std::vector<char> outputBuff(sizeof(device_mgmt_api::get_sp_stats_t), 0);

  auto ret = dm_.serviceRequest(devNum_, device_mgmt_api::DM_CMD::DM_CMD_GET_SP_STATS, nullptr, 0, outputBuff.data(),
                                outputBuff.size(), &hostLatency, &deviceLatency, kDmServiceRequestTimeout);
  if (ret != device_mgmt_api::DM_STATUS_SUCCESS) {
    DV_LOG(ERROR) << "Service request get sp stats failed with return code: " << std::dec << ret;
  } else {
    auto* sp_stats = static_cast<device_mgmt_api::get_sp_stats_t*>(static_cast<void*>(outputBuff.data()));

    spStats_.op.system.power.avg = sp_stats->system_power_avg;
    spStats_.op.system.power.min = sp_stats->system_power_min;
    spStats_.op.system.power.max = sp_stats->system_power_max;
    spStats_.op.system.temperature.avg = sp_stats->system_temperature_avg;
    spStats_.op.system.temperature.min = sp_stats->system_temperature_min;
    spStats_.op.system.temperature.max = sp_stats->system_temperature_max;

    spStats_.op.minion.power.avg = sp_stats->minion_power_avg;
    spStats_.op.minion.power.min = sp_stats->minion_power_min;
    spStats_.op.minion.power.max = sp_stats->minion_power_max;
    spStats_.op.minion.temperature.avg = sp_stats->minion_temperature_avg;
    spStats_.op.minion.temperature.min = sp_stats->minion_temperature_min;
    spStats_.op.minion.temperature.max = sp_stats->minion_temperature_max;
    spStats_.op.minion.voltage.avg = sp_stats->minion_voltage_avg;
    spStats_.op.minion.voltage.min = sp_stats->minion_voltage_min;
    spStats_.op.minion.voltage.max = sp_stats->minion_voltage_max;
    spStats_.op.minion.freq.avg = sp_stats->minion_freq_avg;
    spStats_.op.minion.freq.min = sp_stats->minion_freq_min;
    spStats_.op.minion.freq.max = sp_stats->minion_freq_max;

    spStats_.op.sram.power.avg = sp_stats->sram_power_avg;
    spStats_.op.sram.power.min = sp_stats->sram_power_min;
    spStats_.op.sram.power.max = sp_stats->sram_power_max;
    spStats_.op.sram.temperature.avg = sp_stats->sram_temperature_avg;
    spStats_.op.sram.temperature.min = sp_stats->sram_temperature_min;
    spStats_.op.sram.temperature.max = sp_stats->sram_temperature_max;
    spStats_.op.sram.voltage.avg = sp_stats->sram_voltage_avg;
    spStats_.op.sram.voltage.min = sp_stats->sram_voltage_min;
    spStats_.op.sram.voltage.max = sp_stats->sram_voltage_max;
    spStats_.op.sram.freq.avg = sp_stats->sram_freq_avg;
    spStats_.op.sram.freq.min = sp_stats->sram_freq_min;
    spStats_.op.sram.freq.max = sp_stats->sram_freq_max;

    spStats_.op.noc.power.avg = sp_stats->noc_power_avg;
    spStats_.op.noc.power.min = sp_stats->noc_power_min;
    spStats_.op.noc.power.max = sp_stats->noc_power_max;
    spStats_.op.noc.temperature.avg = sp_stats->noc_temperature_avg;
    spStats_.op.noc.temperature.min = sp_stats->noc_temperature_min;
    spStats_.op.noc.temperature.max = sp_stats->noc_temperature_max;
    spStats_.op.noc.voltage.avg = sp_stats->noc_voltage_avg;
    spStats_.op.noc.voltage.min = sp_stats->noc_voltage_min;
    spStats_.op.noc.voltage.max = sp_stats->noc_voltage_max;
    spStats_.op.noc.freq.avg = sp_stats->noc_freq_avg;
    spStats_.op.noc.freq.min = sp_stats->noc_freq_min;
    spStats_.op.noc.freq.max = sp_stats->noc_freq_max;
  }
}

void EtMetrics::collectDeviceDetails(void) {
  uint32_t hostLatency;
  uint64_t deviceLatency;
  device_mgmt_api::asset_info_t assetInfo = {0};
  auto ret = dm_.serviceRequest(devNum_, device_mgmt_api::DM_CMD_GET_MODULE_PART_NUMBER, nullptr, 0,
                                static_cast<char*>(static_cast<void*>(&assetInfo)), sizeof(assetInfo), &hostLatency,
                                &deviceLatency, kDmServiceRequestTimeout);
  if (ret != device_mgmt_api::DM_STATUS_SUCCESS) {
    DV_LOG(ERROR) << "Service request get asset info failed with return code: " << std::dec << ret;
  } else {
    std::stringstream sstream;
    sstream << *static_cast<uint32_t*>(static_cast<void*>(assetInfo.asset));
    cardId_ = sstream.str();
  }

  device_mgmt_api::firmware_version_t versions = {0};
  ret = dm_.serviceRequest(devNum_, device_mgmt_api::DM_CMD_GET_MODULE_FIRMWARE_REVISIONS, nullptr, 0,
                           static_cast<char*>(static_cast<void*>(&versions)), sizeof(versions), &hostLatency,
                           &deviceLatency, kDmServiceRequestTimeout);
  if (ret != device_mgmt_api::DM_STATUS_SUCCESS) {
    DV_LOG(ERROR) << "Service request get firmware version failed with return code: " << std::dec << ret;
  } else {
    fwVersion_ = formatVersion(versions.fw_release_rev);
    pmicFwVersion_ = formatVersion(versions.pmic_v);
  }
}

void EtMetrics::collectFreqStats(void) {
  uint32_t hostLatency;
  uint64_t deviceLatency;
  auto ret = dm_.serviceRequest(devNum_, device_mgmt_api::DM_CMD_GET_ASIC_FREQUENCIES, nullptr, 0,
                                static_cast<char*>(static_cast<void*>(&freqStats_)), sizeof(freqStats_), &hostLatency,
                                &deviceLatency, kDmServiceRequestTimeout);
  if (ret != device_mgmt_api::DM_STATUS_SUCCESS) {
    DV_LOG(ERROR) << "Service request get asic frequencies failed with return code: " << std::dec << ret;
  }
}

void EtMetrics::collectVoltStats(void) {
  uint32_t hostLatency;
  uint64_t deviceLatency;
  auto ret = dm_.serviceRequest(devNum_, device_mgmt_api::DM_CMD_GET_MODULE_VOLTAGE, nullptr, 0,
                                static_cast<char*>(static_cast<void*>(&moduleVoltStats_)), sizeof(moduleVoltStats_),
                                &hostLatency, &deviceLatency, kDmServiceRequestTimeout);
  if (ret != device_mgmt_api::DM_STATUS_SUCCESS) {
    DV_LOG(ERROR) << "Service request get module voltage failed with return code: " << std::dec << ret;
  }
  ret = dm_.serviceRequest(devNum_, device_mgmt_api::DM_CMD_GET_ASIC_VOLTAGE, nullptr, 0,
                           static_cast<char*>(static_cast<void*>(&asicVoltStats_)), sizeof(asicVoltStats_),
                           &hostLatency, &deviceLatency, kDmServiceRequestTimeout);
  if (ret != device_mgmt_api::DM_STATUS_SUCCESS) {
    DV_LOG(ERROR) << "Service request get asic voltage failed with return code: " << std::dec << ret;
  }
}

void EtMetrics::collectMmStats(void) {
  uint32_t hostLatency;
  uint64_t deviceLatency;
  std::vector<char> outputBuff(sizeof(device_mgmt_api::get_mm_stats_t), 0);

  auto ret = dm_.serviceRequest(devNum_, device_mgmt_api::DM_CMD::DM_CMD_GET_MM_STATS, nullptr, 0, outputBuff.data(),
                                outputBuff.size(), &hostLatency, &deviceLatency, kDmServiceRequestTimeout);

  if (ret != device_mgmt_api::DM_STATUS_SUCCESS) {
    DV_LOG(ERROR) << "Service request get mm stats failed with return code: " << std::dec << ret;
  } else {
    auto* mm_stats = static_cast<device_mgmt_api::get_mm_stats_t*>(static_cast<void*>(outputBuff.data()));

    mmStats_.computeResources.cm_bw.avg = mm_stats->cm_bw_avg;
    mmStats_.computeResources.cm_bw.min = mm_stats->cm_bw_min;
    mmStats_.computeResources.cm_bw.max = mm_stats->cm_bw_max;
    mmStats_.computeResources.cm_utilization.avg = mm_stats->cm_utilization_avg;
    mmStats_.computeResources.cm_utilization.min = mm_stats->cm_utilization_min;
    mmStats_.computeResources.cm_utilization.max = mm_stats->cm_utilization_max;

    mmStats_.computeResources.pcie_dma_read_utilization.avg = mm_stats->pcie_dma_read_utilization_avg;
    mmStats_.computeResources.pcie_dma_read_utilization.min = mm_stats->pcie_dma_read_utilization_min;
    mmStats_.computeResources.pcie_dma_read_utilization.max = mm_stats->pcie_dma_read_utilization_max;

    mmStats_.computeResources.pcie_dma_write_utilization.avg = mm_stats->pcie_dma_write_utilization_avg;
    mmStats_.computeResources.pcie_dma_write_utilization.min = mm_stats->pcie_dma_write_utilization_min;
    mmStats_.computeResources.pcie_dma_write_utilization.max = mm_stats->pcie_dma_write_utilization_max;

    mmStats_.computeResources.pcie_dma_read_bw.avg = mm_stats->pcie_dma_read_bw_avg;
    mmStats_.computeResources.pcie_dma_read_bw.min = mm_stats->pcie_dma_read_bw_min;
    mmStats_.computeResources.pcie_dma_read_bw.max = mm_stats->pcie_dma_read_bw_max;

    mmStats_.computeResources.pcie_dma_write_bw.avg = mm_stats->pcie_dma_write_bw_avg;
    mmStats_.computeResources.pcie_dma_write_bw.min = mm_stats->pcie_dma_write_bw_min;
    mmStats_.computeResources.pcie_dma_write_bw.max = mm_stats->pcie_dma_write_bw_max;

    mmStats_.computeResources.ddr_read_bw.avg = mm_stats->ddr_read_bw_avg;
    mmStats_.computeResources.ddr_read_bw.min = mm_stats->ddr_read_bw_min;
    mmStats_.computeResources.ddr_read_bw.max = mm_stats->ddr_read_bw_max;

    mmStats_.computeResources.ddr_write_bw.avg = mm_stats->ddr_write_bw_avg;
    mmStats_.computeResources.ddr_write_bw.min = mm_stats->ddr_write_bw_min;
    mmStats_.computeResources.ddr_write_bw.max = mm_stats->ddr_write_bw_max;

    mmStats_.computeResources.l2_l3_read_bw.avg = mm_stats->l2_l3_read_bw_avg;
    mmStats_.computeResources.l2_l3_read_bw.min = mm_stats->l2_l3_read_bw_min;
    mmStats_.computeResources.l2_l3_read_bw.max = mm_stats->l2_l3_read_bw_max;

    mmStats_.computeResources.l2_l3_write_bw.avg = mm_stats->l2_l3_write_bw_avg;
    mmStats_.computeResources.l2_l3_write_bw.min = mm_stats->l2_l3_write_bw_min;
    mmStats_.computeResources.l2_l3_write_bw.max = mm_stats->l2_l3_write_bw_max;
  }
}

std::string EtMetrics::formatVersion(uint32_t ver) {
  return "v" + std::to_string((ver >> 24) & 0xff) + "." + std::to_string((ver >> 16) & 0xff) + "." +
         std::to_string((ver >> 8) & 0xff);
}

void EtMetrics::outputCsvHeader(void) {
  output_ << "timestamp,device_id,card_id,etsoc_fw,pmic_fw,cma_allocated_mb,cma_alloc_rate_mb_sec,"
          << "system_power_w,system_temp_c,minion_power_w,minion_temp_c,minion_voltage_mv,minion_freq_mhz,"
          << "sram_power_w,sram_temp_c,sram_voltage_mv,sram_freq_mhz,noc_power_w,noc_temp_c,"
          << "noc_voltage_mv,noc_freq_mhz,cm_utilization_pct,cm_throughput_kernel_sec,"
          << "pcie_dma_read_bw_mbs,pcie_dma_write_bw_mbs,pcie_dma_read_util_pct,"
          << "pcie_dma_write_util_pct,ddr_read_bw_mbs,ddr_write_bw_mbs,l2_l3_read_bw_mbs,"
          << "l2_l3_write_bw_mbs,minion_shire_mhz,noc_mhz,mem_shire_mhz,ddr_mhz,"
          << "pcie_shire_mhz,io_shire_mhz\n";
}

void EtMetrics::outputCsvRow(void) {
  if (!noHeader_ && !headerOutput_) {
    outputCsvHeader();
    headerOutput_ = true;
    if (outputHeaderOnce_) {
      noHeader_ = true;
    }
  }

  auto now = std::chrono::system_clock::now();
  auto epoch = now.time_since_epoch();
  double timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count() / 1000.0;

  const auto& op = spStats_.op;
  double systemPowerW = POWER_10MW_TO_W(op.system.power.avg);
  double minionPowerW = POWER_MW_TO_W(op.minion.power.avg);
  double sramPowerW = POWER_MW_TO_W(op.sram.power.avg);
  double nocPowerW = POWER_MW_TO_W(op.noc.power.avg);
  uint64_t minionVoltage = BIN2VOLTAGE(moduleVoltStats_.minion, 250, 5, 1);
  uint64_t sramVoltage = BIN2VOLTAGE(moduleVoltStats_.l2_cache, 250, 5, 1);
  uint64_t nocVoltage = BIN2VOLTAGE(moduleVoltStats_.noc, 250, 5, 1);

  output_ << std::fixed << std::setprecision(3) << timestamp << "," << devNum_ << "," << cardId_ << "," << fwVersion_
          << "," << pmicFwVersion_ << "," << memStats_.cmaAllocated << "," << memStats_.cmaAllocationRate << ","
          << std::setprecision(2) << systemPowerW << "," << op.system.temperature.avg << "," << minionPowerW << ","
          << op.minion.temperature.avg << "," << minionVoltage << "," << op.minion.freq.avg << "," << std::setprecision(2)
          << sramPowerW << "," << op.sram.temperature.avg << "," << sramVoltage << "," << op.sram.freq.avg << ","
          << std::setprecision(2) << nocPowerW << "," << op.noc.temperature.avg << "," << nocVoltage << ","
          << op.noc.freq.avg << "," << mmStats_.computeResources.cm_utilization.avg << ","
          << mmStats_.computeResources.cm_bw.avg << "," << mmStats_.computeResources.pcie_dma_read_bw.avg << ","
          << mmStats_.computeResources.pcie_dma_write_bw.avg << ","
          << mmStats_.computeResources.pcie_dma_read_utilization.avg << ","
          << mmStats_.computeResources.pcie_dma_write_utilization.avg << ","
          << mmStats_.computeResources.ddr_read_bw.avg << "," << mmStats_.computeResources.ddr_write_bw.avg << ","
          << mmStats_.computeResources.l2_l3_read_bw.avg << "," << mmStats_.computeResources.l2_l3_write_bw.avg << ","
          << freqStats_.minion_shire_mhz << "," << freqStats_.noc_mhz << "," << freqStats_.mem_shire_mhz << ","
          << freqStats_.ddr_mhz << "," << freqStats_.pcie_shire_mhz << "," << freqStats_.io_shire_mhz << "\n";
}

void EtMetrics::outputStats(void) {
  outputCsvRow();
  output_.flush();
}

int main(int argc, char** argv) {
  char* endptr = NULL;
  long int devNum = 0;
  std::chrono::duration delay = std::chrono::milliseconds(kUpdateDelayMS);
  std::string outputFile;
  bool outputHeaderOnce = false;
  bool noHeader = false;
  bool usageError = false;

  if (argc < 2 || argc > 6) {
    usageError = true;
  } else {
    devNum = strtol(argv[1], &endptr, 10);
    if (*endptr || devNum < 0 || devNum > kMaxDeviceNum) {
      usageError = true;
    } else {
      int i = 2;
      if (argc > 2 && argv[i][0] != '-') {
        auto delayCount = strtol(argv[i], &endptr, 10);
        if (*endptr || delayCount < 0) {
          usageError = true;
        } else {
          delay = std::chrono::milliseconds(delayCount);
        }
        i++;
      }

      while (!usageError && i < argc) {
        if (!outputFile.empty()) {
          usageError = true;
        } else if (!strcmp(argv[i], "-o") || !strcmp(argv[i], "--output")) {
          if (++i < argc) {
            outputFile = argv[i];
          } else {
            usageError = true;
          }
        } else if (!outputHeaderOnce && (!strcmp(argv[i], "-1") || !strcmp(argv[i], "--once"))) {
          outputHeaderOnce = true;
        } else if (!noHeader && !strcmp(argv[i], "--no-header")) {
          noHeader = true;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
          std::cout << ET_METRICS " version 1.0.0\n"
                    << "Usage: " << argv[0] << " DEVNO [DELAY_MS] [OPTIONS]\n"
                    << "\n"
                    << "DEVNO       - Device number (0-63)\n"
                    << "DELAY_MS    - Sampling interval in milliseconds (default: 1000)\n"
                    << "OPTIONS:\n"
                    << "  -h, --help           Show help message\n"
                    << "  -o, --output FILE    Output to file instead of stdout (optional)\n"
                    << "  -1, --once           Output header only once, then data rows\n"
                    << "  --no-header          Skip CSV header row\n"
                    << "\n"
                    << "Example:\n"
                    << "  " << argv[0] << " 0 500           # Output to stdout with 500ms sampling\n"
                    << "  " << argv[0] << " 0 100 -o out.csv # Output to file\n"
                    << "  " << argv[0] << " 0 1000 -1        # Output header once then data\n";
          return 0;
        } else {
          usageError = true;
        }
        i++;
      }
    }
  }

  if (usageError) {
    std::cerr << ET_METRICS " version 1.0.0\n"
              << "Usage: " << argv[0] << " DEVNO [DELAY_MS] [OPTIONS]\n"
              << "\n"
              << "DEVNO       - Device number (0-63)\n"
              << "DELAY_MS    - Sampling interval in milliseconds (default: 1000)\n"
              << "OPTIONS:\n"
              << "  -h, --help           Show help message\n"
              << "  -o, --output FILE    Output to file instead of stdout (optional)\n"
              << "  -1, --once           Output header only once, then data rows\n"
              << "  --no-header          Skip CSV header row\n";
    return 1;
  }

  std::string devName = "/dev/et" + std::to_string(devNum) + "_mgmt";
  struct stat buf;
  if (stat(devName.data(), &buf) != 0) {
    std::cerr << devName.data() << " file error: " << std::strerror(errno) << std::endl;
    return 1;
  }

  logging::LoggerDefault loggerDefault_;
  g3::log_levels::disable(DEBUG);
  google::InitGoogleLogging(argv[0]);
  setbuf(stdout, NULL);

  struct sigaction sa;
  sa.sa_handler = signalHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = 0;
  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGTERM, &sa, NULL);

  std::unique_ptr<dev::IDeviceLayer> dl = dev::IDeviceLayer::createPcieDeviceLayer(false, true);
  device_management::DeviceManagement& dm = device_management::DeviceManagement::getInstance(dl.get());

  std::ofstream outFileStream;
  std::ostream* outputStream = &std::cout;
  if (!outputFile.empty()) {
    outFileStream.open(outputFile, std::ios::out | std::ios::trunc);
    if (!outFileStream.is_open()) {
      std::cerr << "Error: unable to open output file " << outputFile << std::endl;
      return 1;
    }
    outputStream = &outFileStream;
  }

  EtMetrics etMetrics(devNum, dl, dm, *outputStream, outputHeaderOnce, noHeader);

  auto checkPoint = std::chrono::steady_clock::now();
  while (!etMetrics.stopStats()) {
    if (checkPoint > std::chrono::steady_clock::now()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } else {
      checkPoint = std::chrono::steady_clock::now() + delay;
      etMetrics.collectStats();
      etMetrics.outputStats();
    }
  }

  if (!outputFile.empty() && outFileStream.is_open()) {
    outFileStream.close();
  }

  google::ShutdownGoogleLogging();
  return 0;
}