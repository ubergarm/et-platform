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

static const std::string KERNELS_DIR = "/lib/esperanto-fw/kernels";

#ifdef ET_INSTALL_DIR
std::string defaultInstallDir = ET_INSTALL_DIR + std::string("/sw-platform-riscv-sysroot/");
#else
std::string defaultInstallDir = "";
#endif

// FIXME: following options are just to speed up  debug-cycle. long-term we should default to void.
DEFINE_string(gp_sdk_device_installdir, defaultInstallDir, "Path to gp-sdk-device installation directory");
std::string defaultKernel = FLAGS_gp_sdk_device_installdir + KERNELS_DIR + "/trace.elf";
// std::string defaultKernel =  "./kernels.elf";

DEFINE_string(kernel_path, defaultKernel, "ET-SoC-1 kernel path and filename");
DEFINE_string(simulator_params, "-l -lm 0", "Hyperparameters to pass to simulator, overrides default values");
DEFINE_string(simulator_installdir, "", "Path to simulator installation directory");
DEFINE_bool(simulator_start_gdb, false, "Enable sysemu gdb");
// TODO "runtime-install-prefix", "num-devices"

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

emu::SysEmuOptions getDefaultOptions() {
  auto [device_bootloader_path, device_minion_rt_path] = getDeviceArtifactsBasePaths();
  const fs::path BOOTROM_TRAMPOLINE_TO_BL2_ELF = device_bootloader_path / "BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
  const fs::path BL2_ELF                       = device_bootloader_path / "ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
  const fs::path MASTER_MINION_ELF             = device_minion_rt_path / "MasterMinion/MasterMinion.elf";
  const fs::path MACHINE_MINION_ELF            = device_minion_rt_path / "MachineMinion/MachineMinion.elf";
  const fs::path WORKER_MINION_ELF             = device_minion_rt_path / "WorkerMinion/WorkerMinion.elf";

  constexpr uint64_t kSysEmuMaxCycles = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t kSysEmuMinionShiresMask = 0x1FFFFFFFFu;

  emu::SysEmuOptions sysEmuOptions;
  sysEmuOptions.bootromTrampolineToBL2ElfPath = BOOTROM_TRAMPOLINE_TO_BL2_ELF;
  sysEmuOptions.spBL2ElfPath = BL2_ELF;
  sysEmuOptions.machineMinionElfPath = MACHINE_MINION_ELF;
  sysEmuOptions.masterMinionElfPath = MASTER_MINION_ELF;
  sysEmuOptions.workerMinionElfPath = WORKER_MINION_ELF;
  sysEmuOptions.executablePath = FLAGS_simulator_installdir + "bin/sys_emu";
  sysEmuOptions.runDir = std::filesystem::current_path();
  sysEmuOptions.maxCycles = kSysEmuMaxCycles;
  sysEmuOptions.minionShiresMask = kSysEmuMinionShiresMask;
  sysEmuOptions.puUart0Path = sysEmuOptions.runDir + "/pu_uart0_tx.log";
  sysEmuOptions.puUart1Path = sysEmuOptions.runDir + "/pu_uart1_tx.log";
  sysEmuOptions.spUart0Path = sysEmuOptions.runDir + "/spio_uart0_tx.log";
  sysEmuOptions.spUart1Path = sysEmuOptions.runDir + "/spio_uart1_tx.log";
  sysEmuOptions.startGdb = FLAGS_simulator_start_gdb;

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

  runtime_ = rt::IRuntime::create(deviceLayer_.get(), options);
  devices_ = runtime_->getDevices();

  for (auto i = 0U; i < static_cast<uint32_t>(deviceLayer_->getDevicesCount()); ++i) {
    defaultStreams_.emplace_back(runtime_->createStream(devices_[i]));
    traceStreams_.emplace_back(runtime_->createStream(devices_[i]));
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
  auto abortedKernelHandler = [this](rt::EventId id, std::byte const* context, size_t size,
                                     std::function<void()> freeResources) {
    std::cout << "abortedKernelHandler"
              << " () rt reports that a kernel has been aborted (EventId: " << static_cast<int>(id) << ")\n";
    // TODO: complete abort management. leverage glow coredump infra
    kernelAbort_++;
  };

  runtime_->setOnStreamErrorsCallback(streamErrorHandler);
  runtime_->setOnKernelAbortedErrorCallback(abortedKernelHandler);

  // load the kernel on the device.
  kernel_ = loadKernel(FLAGS_kernel_path);
}

void GenericLauncher::deInitialize() {
  runtime_->unloadCode(kernel_);
}

void GenericLauncher::tearDown() {
  auto timeout = std::chrono::seconds(1);
  for (auto s : defaultStreams_) {
    auto success = runtime_->waitForStream(s, timeout);
    if (!success) {
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
  runtime_.reset();
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
  // all
  //    return {0x1FFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
  // zero
  return {0x100000001ULL, 0x0000000000000001ULL};
}

// TODO: DRY. get this magic number from rutnime/ device headers if possible.
// 4096 bytes per hart, 2048 harts....
constexpr size_t kTraceBytesPerHart = 4096;
constexpr size_t kNumHarts = 2048; // why not 2080?
constexpr size_t kTraceBufferSize = kTraceBytesPerHart * kNumHarts;
// TODO: consolidate configuration.
constexpr bool enableKernelTraces = true;

// TODO: make class fn.
std::optional<rt::UserTrace> getKernelTraceParams(std::byte* deviceTraceBuffer, size_t deviceTraceBufferSize) {
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

void GenericLauncher::dumpTracesToFile(uint64_t fileIdx) {
  if (not enableKernelTraces) {
    return;
  }
  // geting device traces.
  std::vector<std::byte> deviceTrace(kTraceBufferSize);
  runtime_->memcpyDeviceToHost(traceStreams_[devIdx_], traceDeviceBuffer_, deviceTrace.data(), deviceTrace.size());
  // serialize traces to disk
  auto tracesTimeout = std::chrono::seconds(10);
  auto success = runtime_->waitForStream(traceStreams_[devIdx_], tracesTimeout);

  // traces have been copied.. we can remove them from device.
  // FIXME: decouple removal from here to reuse trace-buffer across kernel launches.
  runtime_->freeDevice(devices_[devIdx_], traceDeviceBuffer_);

  if (!success) {
    std::cout << __func__ << "() timeout extracting traces from device\n";
    return;
  }

  auto tracePath =
    std::filesystem::current_path() /
    std::filesystem::path("traceKernels_dev" + std::to_string(devIdx_) + "_" + std::to_string(fileIdx) + ".bin");
  auto traceStream = std::ofstream(tracePath, std::ios::binary | std::ios::out);
  traceStream.write((char*)deviceTrace.data(), deviceTrace.size());
}

void GenericLauncher::waitKernelCompletion(std::chrono::seconds timeout) {
  auto success = runtime_->waitForStream(defaultStreams_[devIdx_], timeout);
  if (success) {
    return;
  }
  // Kernel did not complete on the expected time. let's abort the stream in which
  // the kernel is running.
  std::cout << "[TIMEOUT] " << __func__ << "() Wait for Stream command exceeded " << std::dec << int(timeout.count())
            << " seconds.  Aborting stream\n";
  auto event = runtime_->abortStream(defaultStreams_[devIdx_]);
  // Wait for the abort to complete.
  auto abortTimeout = std::chrono::seconds(10);
  success = runtime_->waitForEvent(event, abortTimeout);
  if (success) {
    std::cout << "[        ] " << __func__ << "() stream aborted correctly \n" << int(defaultStreams_[devIdx_]) << "\n";
    return;
  }

  // could not complete the abort.
  std::cout << "[TIMEOUT] " << __func__ << "() timeout aborting stream \n" << int(defaultStreams_[devIdx_]) << "\n";
  // TODO: we failed to abort the stream. place any mitigation / defensive code here.
}

void GenericLauncher::kernelLaunch() {
  // TODO   make shire-mask config-aware.
  // Alloc space on device to get user traces. TODO: split into prep work so we can leverage across launches.
  if (enableKernelTraces) {
    traceDeviceBuffer_ = runtime_->mallocDevice(devices_[devIdx_], kTraceBufferSize);
  }
  std::optional<rt::UserTrace> optUserTrace = getKernelTraceParams(traceDeviceBuffer_, kTraceBufferSize);
  constexpr bool barrier = true;
  constexpr bool flushL3 = false;
  constexpr uint64_t shireMask = 0x1ffffffff;
  runtime_->kernelLaunch(defaultStreams_[devIdx_], kernel_, kernelArgs_, kernelArgsSize_, shireMask, barrier, flushL3,
                         optUserTrace);
}
