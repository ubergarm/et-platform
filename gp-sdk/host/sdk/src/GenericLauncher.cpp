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
#include <iostream>
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

#include "GenericLauncher.h"
#include "llvm/Support/CommandLine.h"
namespace {
static const std::string KERNELS_DIR = "/sw-platform-riscv-sysroot/lib/esperanto-fw/kernels";

llvm::cl::OptionCategory GeneralCat("General Options");
llvm::cl::OptionCategory SysemuCat("Sysemu Options");

llvm::cl::opt<Mode> deviceType{
  "device-type", llvm::cl::desc("Device Type to be used"),
  llvm::cl::values(clEnumValN(Mode::SYSEMU, "sysemu", "ET-SoC-1 software instruction simulator"),
                   clEnumValN(Mode::FAKE, "fake", "Loopback fake devoce"),
                   clEnumValN(Mode::PCIE, "silicon", "Single ETSOC-1 SOC")),
  llvm::cl::init(Mode::SYSEMU), llvm::cl::cat(GeneralCat)};

// FIXME: following options are just to speed up  debug-cycle. long-term we should default to void.
std::string defaultKernel = ET_INSTALL_DIR + KERNELS_DIR + "/trace.elf";
// std::string defaultKernel =  "./kernels.elf";
std::string defaultSimParams = "-l -lm 0";

const llvm::cl::opt<std::string> kernelPath("kernelPath", llvm::cl::desc("ET-SoC-1 kernel and filename"),
                                            llvm::cl::init(defaultKernel), llvm::cl::cat(GeneralCat));

const llvm::cl::opt<std::string>
  simParams("simulator-params", llvm::cl::desc("Hyperparameters to pass to simulator, overrides default values"),
            llvm::cl::init(defaultSimParams), llvm::cl::cat(SysemuCat));

// TODO "runtime-install-prefix", "num-devices"
} // namespace

emu::SysEmuOptions getDefaultOptions() {

  static const std::string BOOTROM_TRAMPOLINE_TO_BL2_ELF =
    "/sw-platform-riscv-sysroot/lib/esperanto-fw/BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
  static const std::string BL2_ELF =
    "/sw-platform-riscv-sysroot/lib/esperanto-fw/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
  static const std::string MASTER_MINION_ELF =
    "/sw-platform-riscv-sysroot/lib/esperanto-fw/MasterMinion/MasterMinion.elf";
  static const std::string MACHINE_MINION_ELF =
    "/sw-platform-riscv-sysroot/lib/esperanto-fw/MachineMinion/MachineMinion.elf";
  static const std::string WORKER_MINION_ELF =
    "/sw-platform-riscv-sysroot/lib/esperanto-fw/WorkerMinion/WorkerMinion.elf";
  static const std::string SYSEMU_INSTALL_DIR = "";

  constexpr uint64_t kSysEmuMaxCycles = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t kSysEmuMinionShiresMask = 0x1FFFFFFFFu;

  emu::SysEmuOptions sysEmuOptions;
  sysEmuOptions.bootromTrampolineToBL2ElfPath = std::string(ET_INSTALL_DIR) + BOOTROM_TRAMPOLINE_TO_BL2_ELF;
  sysEmuOptions.spBL2ElfPath = std::string(ET_INSTALL_DIR) + BL2_ELF;
  sysEmuOptions.machineMinionElfPath = std::string(ET_INSTALL_DIR) + MACHINE_MINION_ELF;
  sysEmuOptions.masterMinionElfPath = std::string(ET_INSTALL_DIR) + MASTER_MINION_ELF;
  sysEmuOptions.workerMinionElfPath = std::string(ET_INSTALL_DIR) + WORKER_MINION_ELF;
  sysEmuOptions.executablePath = std::string(SYSEMU_INSTALL_DIR) + "sys_emu";
  sysEmuOptions.runDir = std::filesystem::current_path();
  sysEmuOptions.maxCycles = kSysEmuMaxCycles;
  sysEmuOptions.minionShiresMask = kSysEmuMinionShiresMask;
  sysEmuOptions.puUart0Path = sysEmuOptions.runDir + "/pu_uart0_tx.log";
  sysEmuOptions.puUart1Path = sysEmuOptions.runDir + "/pu_uart1_tx.log";
  sysEmuOptions.spUart0Path = sysEmuOptions.runDir + "/spio_uart0_tx.log";
  sysEmuOptions.spUart1Path = sysEmuOptions.runDir + "/spio_uart1_tx.log";
  sysEmuOptions.startGdb = false;

  // Pass the sysemu parameters from command line
  auto cmd = simParams.getValue();
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
  switch (mode_) {
  case Mode::PCIE:
    std::cout << "Running tests with PCIE deviceLayer";
    deviceLayer_ = dev::IDeviceLayer::createPcieDeviceLayer();
    break;
  case Mode::SYSEMU: {
    std::cout << "Running tests with SYSEMU deviceLayer";
    auto opts = getDefaultOptions();
    std::vector<decltype(opts)> vopts;
    for (auto i = 0; i < numDevices_; ++i) {
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
  }
  runtime_ = rt::IRuntime::create(deviceLayer_.get(), options);
  devices_ = runtime_->getDevices();

  for (auto i = 0U; i < static_cast<uint32_t>(deviceLayer_->getDevicesCount()); ++i) {
    defaultStreams_.emplace_back(runtime_->createStream(devices_[i]));
    traceStreams_.emplace_back(runtime_->createStream(devices_[i]));
  }

  // Program callbacks for error management.
  auto streamErrorHandler = [&]([[maybe_unused]] rt::EventId id, const rt::StreamError& error) {
    // TO IMPROVE: Currently we don't have the deviceId related to this error.
    std::cout << "streamErrorHandler "
              << "() rt reports an error on a stream command(EventId: " << static_cast<int>(id) << "):\n"
              << error.getString();
    if ((error.errorCode_ == rt::DeviceErrorCode::DmaHostAborted) or
        (error.errorCode_ == rt::DeviceErrorCode::KernelLaunchHostAborted)) {
      std::cout << std::to_string(error.errorCode_) << " Errors during aborts are expected, ignoring";
      return;
    }
  };

  // Program callback when we want kernel aborts (due to a timeout) to dump corefiles
  auto abortedKernelHandler = [](rt::EventId id, std::byte const* context, size_t size,
                                 std::function<void()> freeResources) {
    std::cout << "abortedKernelHandler"
              << " () rt reports that a kernel has been aborted (EventId: " << static_cast<int>(id) << ")\n";
    // TODO: complete abort management. leverage glow coredump infra
  };

  runtime_->setOnStreamErrorsCallback(streamErrorHandler);
  runtime_->setOnKernelAbortedErrorCallback(abortedKernelHandler);

  // load the kernel on the device.
  kernel_ = loadKernel(kernelPath.getValue());
}

void GenericLauncher::deInitialize() {
  runtime_->unloadCode(kernel_);
}

void GenericLauncher::tearDown() {
  for (auto s : defaultStreams_) {
    runtime_->waitForStream(s);
    runtime_->destroyStream(s);
  }
  for (auto s : traceStreams_) {
    runtime_->waitForStream(s);
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
  std::cout << __func__ << "() kernnel " << int(res.kernel_) << " loaded at " << std::hex << res.loadAddress_ << "\n";
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

void GenericLauncher::dumpTracesToFile() {
  if (not enableKernelTraces) {
    return;
  }
  // geting device traces.
  std::vector<std::byte> deviceTrace(kTraceBufferSize);
  runtime_->memcpyDeviceToHost(traceStreams_[devIdx_], traceDeviceBuffer_, deviceTrace.data(), deviceTrace.size());
  // serialize traces to disck
  runtime_->waitForStream(traceStreams_[devIdx_]);
  // traces have been copied.. we can remove them from device.
  runtime_->freeDevice(devices_[devIdx_], traceDeviceBuffer_);

  auto tracePath =
    std::filesystem::current_path() / std::filesystem::path("traceKernels_dev" + std::to_string(devIdx_) + ".bin");
  auto traceStream = std::ofstream(tracePath, std::ios::binary | std::ios::out);
  traceStream.write((char*)deviceTrace.data(), deviceTrace.size());
}

void GenericLauncher::waitKernelCompletion(std::chrono::seconds timeout) {
  // TODO: need a specific wait from GenericLauncher PoV
  auto success = runtime_->waitForStream(defaultStreams_[devIdx_], timeout);
  if (success) {
    return;
  }

  // Kernel did not complete on the expected time. let's abort the stream in which
  // the ckernel is running.
  std::cout << "[TIMEOUT] " << __func__ << "() Wait for Stream command exceeded " << int(timeout.count())
            << " seconds.  Aborting stream\n";
  auto event = runtime_->abortStream(defaultStreams_[devIdx_]);
  // Wait for the abort to complete. (runtime will be in aborting state
  success = runtime_->waitForEvent(event);
  if (success) {
    std::cout << "[        ] " << __func__ << "() stream aborted correctly " << int(defaultStreams_[devIdx_]) << "\n";
    return;
  }

  // could not complete the abort.
  std::cout << "[TIMEOUT] " << __func__ << "() timeout aborting stream " << int(defaultStreams_[devIdx_]) << "\n";
  // TODO: we failed to abort the stream. place any mitigation / defensive code here.
}

void GenericLauncher::kernelLaunch() {
  // TODO   make shire-mask config-aware.
  // Alloc space on device to get user traces. TODO: split into prep work so we can leverae across launches.
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
