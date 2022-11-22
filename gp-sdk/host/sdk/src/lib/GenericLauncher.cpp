//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#include <cassert>
#include <esperanto/et-trace/encoder.h>
#include <fstream>
#include <gflags/gflags.h>
#include <iostream>
#include <iterator>
#include <runtime/DeviceLayerFake.h>
#include <sw-sysemu/SysEmuOptions.h>

#include <cstdint>
#include <tuple>
#include <unistd.h>

#if __has_include("filesystem")
#include <filesystem>
#elif __has_include("experimental/filesystem")
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#endif
namespace fs = std::filesystem;

#include "GenericLauncher.h"
#include "RuntimeImpWithCoreDump.h"

DEFINE_string(gp_sdk_device_installdir, "", "Path to gp-sdk-device installation directory");

DEFINE_string(simulator_params, "", "Hyperparameters to pass to simulator, overrides default values");

DEFINE_bool(enableCoreDump, false, "Enable core dump");

// Trace Buffer realted constants.
constexpr size_t kTraceBytesPerHart = 4096;
constexpr size_t kNumHarts = 2048; 
constexpr size_t kTraceBufferSize = kTraceBytesPerHart * kNumHarts;
constexpr bool enableKernelTraces = true;

#ifdef WITH_SYSEMU_PATHS
std::tuple<fs::path, fs::path> getDeviceArtifactsBasePaths() {
  fs::path device_bootloader_path = FLAGS_gp_sdk_device_installdir;
  fs::path device_minion_rt_path = FLAGS_gp_sdk_device_installdir;
#ifdef WITH_CONAN
  device_bootloader_path /= "device-bootloaders";
  device_minion_rt_path /= "device-minion-runtime";
#endif
  fs::path postfix = "lib/esperanto-fw";
  device_bootloader_path /= postfix;
  device_minion_rt_path /= postfix;
  return {device_bootloader_path, device_minion_rt_path};
}
#endif

emu::SysEmuOptions getDefaultOptions() {
#ifdef WITH_SYSEMU_PATHS
  auto [device_bootloader_path, device_minion_rt_path] = getDeviceArtifactsBasePaths();
  const fs::path BOOTROM_TRAMPOLINE_TO_BL2_ELF =
    device_bootloader_path / "BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
  const fs::path BL2_ELF = device_bootloader_path / "ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
  const fs::path MASTER_MINION_ELF = device_minion_rt_path / "MasterMinion/MasterMinion.elf";
  const fs::path MACHINE_MINION_ELF = device_minion_rt_path / "MachineMinion/MachineMinion.elf";
  const fs::path WORKER_MINION_ELF = device_minion_rt_path / "WorkerMinion/WorkerMinion.elf";
#endif
  constexpr uint64_t kSysEmuMaxCycles = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t kSysEmuMinionShiresMask = 0x1FFFFFFFFu;

  emu::SysEmuOptions sysEmuOptions;
#ifdef WITH_SYSEMU_PATHS
  sysEmuOptions.bootromTrampolineToBL2ElfPath = BOOTROM_TRAMPOLINE_TO_BL2_ELF;
  sysEmuOptions.spBL2ElfPath = BL2_ELF;
  sysEmuOptions.machineMinionElfPath = MACHINE_MINION_ELF;
  sysEmuOptions.masterMinionElfPath = MASTER_MINION_ELF;
  sysEmuOptions.workerMinionElfPath = WORKER_MINION_ELF;
#endif
  sysEmuOptions.runDir = std::filesystem::current_path();
  sysEmuOptions.maxCycles = kSysEmuMaxCycles;
  sysEmuOptions.minionShiresMask = kSysEmuMinionShiresMask;
  sysEmuOptions.puUart0Path = sysEmuOptions.runDir + "/pu_uart0_tx.log";
  sysEmuOptions.puUart1Path = sysEmuOptions.runDir + "/pu_uart1_tx.log";
  sysEmuOptions.spUart0Path = sysEmuOptions.runDir + "/spio_uart0_tx.log";
  sysEmuOptions.spUart1Path = sysEmuOptions.runDir + "/spio_uart1_tx.log";
  sysEmuOptions.startGdb = false;

  // Pass the sysemu parameters from command line
  auto cmd = FLAGS_simulator_params;
  std::istringstream iss{cmd};
  sysEmuOptions.additionalOptions =
    std::vector<std::string>{std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{}};

  return sysEmuOptions;
}

std::vector<std::byte> GenericLauncher::readFile(const std::string& path) {
  auto file = std::ifstream(path, std::ios_base::binary);
  if (!file.is_open()) {
    std::cout << __func__ << "kernel file " << path << " not found\n";
    return {};
  }
  auto size = std::filesystem::file_size(path);
  std::vector<std::byte> fileContent(size);
  file.read(reinterpret_cast<char*>(fileContent.data()), size);
  return fileContent;
}

void GenericLauncher::initialize() {

  auto options = rt::getDefaultOptions();
  switch (config_.mode_) {
  case Mode::PCIE:
    std::cout << "Running tests with PCIE deviceLayer";
    deviceLayer_ = dev::IDeviceLayer::createPcieDeviceLayer();
    break;
  case Mode::SYSEMU: {
    std::cout << "Running tests with SYSEMU deviceLayer";
    auto opts = getDefaultOptions();
    std::vector<decltype(opts)> vopts;
    for (auto i = 0; i < config_.numDevices_; ++i) {
      vopts.emplace_back(opts);
      vopts.back().logFile += std::to_string(i);
    }
    deviceLayer_ = dev::IDeviceLayer::createSysEmuDeviceLayer(vopts);
    break;
  }
  case Mode::FAKE:
    std::cout << "Running tests with FAKE deviceLayer";
    deviceLayer_ = std::make_unique<dev::DeviceLayerFake>();
    options.checkDeviceApiVersion_ = false;
    break;
  case Mode::LAST:
    std::cout << "Unsupported device \n";
    exit(-1);
    break;
  }

  createRuntime(FLAGS_enableCoreDump, options);

  devices_ = runtime_->getDevices();

  for (auto i = 0U; i < static_cast<uint32_t>(deviceLayer_->getDevicesCount()); ++i) {
    defaultStreams_.emplace_back(runtime_->createStream(devices_[i]));
    traceStreams_.emplace_back(runtime_->createStream(devices_[i]));

    abortManager_.registerStream(defaultStreams_[i], devices_[i]);
  }

  // Program callbacks for error management.
  auto streamErrorHandler = [&, this]([[maybe_unused]] rt::EventId id, const rt::StreamError& error) {
    // TO IMPROVE: Currently we don't have the deviceId related to this error.
    std::cout << "streamErrorHandler "
              << "() rt reports an error on a stream command(EventId: " << static_cast<int>(id) << "):\n"
              << error.getString();
    if ((error.errorCode_ == rt::DeviceErrorCode::DmaHostAborted) or
        (error.errorCode_ == rt::DeviceErrorCode::KernelLaunchHostAborted)) {
      std::cout << std::to_string(error.errorCode_) << " Errors during aborts are expected, ignoring";
      return;
    }
    kernelError_++;
  };

  // Program callback when we want kernel aborts (due to a timeout) to dump corefiles
  auto abortedKernelHandler = [this, rt = runtime_.get()](rt::EventId id, std::byte const* context, size_t size,
                                                          std::function<void()> freeResources) {
    std::cout << "abortedKernelHandler"
              << " () rt reports that a kernel has been aborted (EventId: " << static_cast<int>(id) << ")\n";
    kernelAbort_++;

    // Wait until the kernel has been fully aborted so that further commands
    // are not aborted
    abortManager_.waitKernelLaunchAborted(id);
    if (FLAGS_enableCoreDump) {
      // Retrieve the error context from the device and handle the error
      auto error = abortManager_.retrieveErrorContext(rt, id, context, size);
      if (error.has_value()) {
        reportUserException(error.value());
        CoreDumper::dumpCore(abortManager_, rt, id, error.value());
      } else {
        LOG(WARNING) << "Device error (core dump not enabled)";
      }
    }
    else {
      LOG(WARNING) << "Device error (core dump not enabled)";
    }

    // Release the runtime resources before allowing the aborter thread to
    // continue
    freeResources();

    // Allow the aborter thread to continue
    abortManager_.notifyDeviceAbortCallback(id);

    LOG(FATAL) << "Kernel aborted. GP SDK cannot recover from this, "
                  "finishing the execution";
  };

  runtime_->setOnStreamErrorsCallback(streamErrorHandler);
  runtime_->setOnKernelAbortedErrorCallback(abortedKernelHandler);

  // Alloc space on device for user traces. Note: This buffer will be reused across differnet kernel launches.
  if (enableKernelTraces) {
    traceDeviceBuffer_ = runtime_->mallocDevice(devices_[devIdx_], kTraceBufferSize);
  }
}

void GenericLauncher::unLoadKernel(rt::KernelId kernelId) {
  runtime_->unloadCode(kernelId);
}

void GenericLauncher::tearDown() { 

  if(enableKernelTraces) {
    runtime_->freeDevice(devices_[devIdx_], traceDeviceBuffer_);
  }

  auto timeout = std::chrono::seconds(1);
  for (auto s : defaultStreams_) {
    auto success = runtime_->waitForStream(s, timeout);
    if (success) {
      abortManager_.clearKernelLaunches(s);
    } else {
      std::cout << __func__ << "() default stream " << uint32_t(s) << " wait timeout\n";
    }
    runtime_->destroyStream(s);
  }
  for (auto s : traceStreams_) {
    auto success = runtime_->waitForStream(s, timeout);
    if (!success) {
      std::cout << __func__ << "() traces stream " << uint32_t(s) << " wait timeout\n";
    }
    runtime_->destroyStream(s);
  }

  resetRuntime(FLAGS_enableCoreDump);
  defaultStreams_.clear();
  traceStreams_.clear();
  devices_.clear();
  deviceLayer_.reset();
}

rt::KernelId GenericLauncher::loadKernel(const std::string& kernelName, uint32_t deviceIdx) {
  auto kernelContent = readFile(kernelName);
  if (kernelContent.empty()) {
    exit(-1);
  }
  assert(devices_.size() > deviceIdx);
  auto st = defaultStreams_[deviceIdx];
  auto res = runtime_->loadCode(st, kernelContent.data(), kernelContent.size());
  runtime_->waitForEvent(res.event_);
  std::cout << __func__ << "() kernel " << int(res.kernel_) << " loaded at " << std::hex << res.loadAddress_ << "\n";

  return res.kernel_;
}

// TODO: make it configuraion-aware.
std::tuple<uint64_t, uint64_t> getTraceMinions() {
  // all (shireMask: threadMask)
  return {0xFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
}



std::optional<rt::UserTrace> fillKernelTraceParams(std::byte* deviceTraceBuffer, size_t deviceTraceBufferSize) {
  if (not enableKernelTraces) {
    return std::nullopt;
  }

  rt::UserTrace traceParams;
  auto [shireMask, threadMask] = getTraceMinions();
  traceParams.buffer_ = uint64_t(deviceTraceBuffer);
  traceParams.buffer_size_ = deviceTraceBufferSize;
  traceParams.shireMask_ = shireMask;
  traceParams.threadMask_ = threadMask;
  traceParams.eventMask_ = TRACE_EVENT_ENABLE_ALL;
  traceParams.filterMask_ = TRACE_FILTER_ENABLE_ALL;
  return traceParams;
}

void GenericLauncher::dumpTracesToFile(uint64_t fileIdx, rt::KernelId kernelId) {
  if (not enableKernelTraces) {
    return;
  }
  // geting device traces.
  std::vector<std::byte> deviceTrace(kTraceBufferSize);
  runtime_->memcpyDeviceToHost(traceStreams_[devIdx_], traceDeviceBuffer_, deviceTrace.data(), deviceTrace.size());
  auto tracesTimeout = std::chrono::seconds(10);
  auto success = runtime_->waitForStream(traceStreams_[devIdx_], tracesTimeout);

  if (!success) {
    std::cout << __func__ << "() timeout extracting traces from device\n";
    return;
  }

  // serialize traces to disk
  std::string traceSuffix = "";
  if (int(kernelId) != -1) {
    traceSuffix = "_" + std::to_string(int(kernelId));
  }
  
  auto tracePath =
    std::filesystem::current_path() /
    std::filesystem::path("traceKernels_dev" + std::to_string(devIdx_) + "_" + std::to_string(fileIdx) + traceSuffix + ".bin");
  auto traceStream = std::ofstream(tracePath, std::ios::binary | std::ios::out);
  traceStream.write((char*)deviceTrace.data(), deviceTrace.size());
}

void GenericLauncher::waitKernelCompletion(std::chrono::seconds timeout) {
  auto success = runtime_->waitForStream(defaultStreams_[devIdx_], timeout);

  if (success) {
    abortManager_.clearKernelLaunches(defaultStreams_[devIdx_]);
    return;
  }
  // Kernel did not complete on the expected time. let's abort the stream in which
  // the kernel is running.
  std::cout << "[TIMEOUT] " << __func__ << "() Wait for Stream command exceeded " << std::dec << int(timeout.count())
            << " seconds.  Aborting stream\n";

  std::vector<rt::EventId> abortedKernelLaunchEventIds;
  abortedKernelLaunchEventIds =
    abortManager_.prepareKernelLaunchAbort(devices_[devIdx_], defaultStreams_[devIdx_], *runtime_);
  auto event = runtime_->abortStream(defaultStreams_[devIdx_]);
  // Wait for the abort to complete.
  auto abortTimeout = std::chrono::seconds(10);
  success = runtime_->waitForEvent(event, abortTimeout);
  if (success) {
    // Allow the callbacks to proceed
    for (auto const& kernelEventId : abortedKernelLaunchEventIds) {
      abortManager_.notifyKernelLaunchAborted(kernelEventId);
    }

    // Wait until the callbacks have been received to avoid destructing the
    // runtime too early
    for (auto const& kernelEventId : abortedKernelLaunchEventIds) {
      abortManager_.waitDeviceAbortCallback(kernelEventId);
    }
    std::cout << "[        ] " << __func__ << "() stream aborted correctly \n" << int(defaultStreams_[devIdx_]) << "\n";
    return;
  }

  // could not complete the abort.
  std::cout << "[TIMEOUT] " << __func__ << "() timeout aborting stream \n" << int(defaultStreams_[devIdx_]) << "\n";
  // TODO: we failed to abort the stream. place any mitigation / defensive code here.
}

void GenericLauncher::doKernelLaunch(rt::KernelId kernelId, std::byte * params, size_t size, uint64_t shireMask) {
  // This promise is used to avoid a race condition where the RT responds with
  // an abort before the core dumper has registered the event id
  std::promise<rt::EventId> promisedEventId;
  promisedEventId = abortManager_.registerKernelLaunch(defaultStreams_[devIdx_], kernel_);

  std::optional<rt::UserTrace> optUserTrace = fillKernelTraceParams(traceDeviceBuffer_, kTraceBufferSize);
  constexpr bool barrier = true;
  constexpr bool flushL3 = false;
  auto eventId = runtime_->kernelLaunch(defaultStreams_[devIdx_], kernelId, params, size, shireMask, barrier, flushL3,
                         optUserTrace);

  promisedEventId.set_value(eventId);
}

void GenericLauncher::reportUserException(const rt::StreamError& error) const {
  LOG(INFO) << "Exception found, need to dump the execution context";
  // Dump execution context into a file
  auto path = std::experimental::filesystem::current_path();
  auto filename = "device_execution_context.txt";
  std::ofstream out(path / filename);
  out << error.getString();
  out << "\n---\n";
  out.close();
}

void GenericLauncher::createRuntime(bool enableCoreDump, rt::Options options) {
  if (enableCoreDump) {
    runtimeBase_ = rt::IRuntime::create(deviceLayer_.get(), options);
    runtime_ = std::make_unique<RuntimeImpWithCoreDump>(runtimeBase_.get(), &abortManager_);
  } else {
    runtime_ = rt::IRuntime::create(deviceLayer_.get(), options);
  }
}

void GenericLauncher::resetRuntime(bool enableCoreDump) {
    if (enableCoreDump) {
    runtimeBase_.reset();
  }
  runtime_.reset();
}

