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
#include <device-layer/IDeviceLayer.h>
#include <esperanto/et-trace/encoder.h>
#include <fstream>
#include <hostUtils/logging/Logger.h>
#include <ios>
#include <iostream>
#include <iterator>
#include <random>
#include <runtime/DeviceLayerFake.h>
#include <runtime/IRuntime.h>
#include <string>
#include <sw-sysemu/SysEmuOptions.h>

#if __has_include("filesystem")
#include <filesystem>
#elif __has_include("experimental/filesystem")
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#endif

#include "llvm/Support/CommandLine.h"

namespace {
enum class Mode { PCIE, SYSEMU, FAKE };
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

inline std::vector<std::byte> readFile(const std::string& path) {
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

std::string to_string(Mode value) {
  std::string result;
  switch (value) {
  case Mode::PCIE:
    result = "Hardware";
    break;
  case Mode::SYSEMU:
    result = "Emulator";
    break;
  case Mode::FAKE:
    result = "Dry run";
    break;
  default:
    result = "Unknown";
    break;
  }
  return result;
}

class Config {
public:
  void setMode(Mode value) {
    mode_ = value;
  }
  Mode getMode() const {
    return mode_;
  }
  void setRuntimeInstallPrefix(const std::string& value) {
    runtimeInstallPrefix_ = value;
  }
  const std::string& getRuntimeInstallPrefix() const {
    return runtimeInstallPrefix_;
  }
  void setNumDevices(size_t value) {
    numDevices_ = value;
  }
  size_t getNumDevices() const {
    return numDevices_;
  }
  void dump() const {
    std::cout << "Selected options:\n";
    std::cout << "  Device type: " << to_string(mode_) << "\n";
    std::cout << "  Runtime install prefix: " << runtimeInstallPrefix_ << "\n";
    std::cout << "  Device count: " << numDevices_ << "\n";
  }

private:
  Mode mode_{Mode::SYSEMU};
  std::string runtimeInstallPrefix_{ET_INSTALL_DIR};
  size_t numDevices_{1};
};

class Test {
public:
  Test() = delete;
  Test(const Config& config)
    : config_{config} {
  }

  void setup() {
    auto options = rt::getDefaultOptions();
    switch (sMode) {
    case Mode::PCIE:
      std::cout << "Running tests with PCIE deviceLayer";
      deviceLayer_ = dev::IDeviceLayer::createPcieDeviceLayer();
      break;
    case Mode::SYSEMU: {
      std::cout << "Running tests with SYSEMU deviceLayer";
      auto opts = getDefaultOptions();
      std::vector<decltype(opts)> vopts;
      for (auto i = 0; i < sNumDevices; ++i) {
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
    }

    // Program callback when stream faces an error
    auto streamErrorHandler = [&]([[maybe_unused]] rt::EventId id, const rt::StreamError& error) {
      // TO IMPROVE: Currently we don't have the deviceId related to this error
      std::cout << "rt reports an error on a stream command"
                << " (EventId: " << static_cast<int>(id) << "):\n"
                << error.getString();
      if ((error.errorCode_ == rt::DeviceErrorCode::DmaHostAborted) or
          (error.errorCode_ == rt::DeviceErrorCode::KernelLaunchHostAborted)) {
        std::cout << std::to_string(error.errorCode_) << " Errors during aborts are expected, ignoring";
        return;
      }

#if 0
    // In silicon we wait until timeout to avoid issues when AMOs are in flight.
    // Since in Emu, the timeout can he high and the issue can be debugged by
    // using the log, here we just abort instead of waiting for the timeout.
    if (deviceType == glow::runtime::DeviceType::Emu) {
      std::cout  << "Cannot get error context in SysEmu. Use SysEmu logging instead.";
      std::terminate();
    }
#endif

      // Do not abort yet. Instead, let the kernel time out. This should minimize
      // in flight AMOs.
    };

    runtime_->setOnStreamErrorsCallback(streamErrorHandler);

    // Program callback when we want kernel aborts (due to a timeout) to dump core
    // files
    auto abortedKernelHandler = [](rt::EventId id, std::byte const* context, size_t size,
                                   std::function<void()> freeResources) {
      std::cout << "rt reports that a kernel has been aborted"
                << " (EventId: " << static_cast<int>(id) << ")\n";
      std::terminate();

#if 0
        // Wait until the kernel has been fully aborted so that further commands
        // are not aborted
        abortManager_.waitKernelLaunchAborted(id);

        // Retrieve the error context from the device and handle the error
        auto error = abortManager_.retrieveErrorContext(rt, id, context, size);
        if (error.has_value()) {
          reportUserException(error.value());
          if (enableCoreDump) {
            ETSOCCoreDumper::dumpCore(abortManager_, rt, id, error.value());
          } else {
            LOG(WARNING) << "Device error (core dump not enabled)";
          }
        }

        // Release the runtime resources before allowing the aborter thread to
        // continue
        freeResources();

        // Allow the aborter thread to continue
        abortManager_.notifyDeviceAbortCallback(id);

        LOG(FATAL) << "Kernel aborted. Glow cannot recover from this, "
                      "finishing the execution";
#endif
    };

    runtime_->setOnKernelAbortedErrorCallback(abortedKernelHandler);
  }

  virtual void run() = 0;

  void tearDown() {
    for (auto s : defaultStreams_) {
      runtime_->waitForStream(s);
      runtime_->destroyStream(s);
    }
    runtime_.reset();
    defaultStreams_.clear();
    devices_.clear();
    deviceLayer_.reset();
  }

  rt::KernelId loadKernel(const std::string& kernelName, uint32_t deviceIdx = 0) {
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

  inline static Mode sMode = Mode::SYSEMU;
  inline static uint8_t sNumDevices = 1;

protected:
  const Config& config_;
  logging::LoggerDefault loggerDefault_;
  std::unique_ptr<dev::IDeviceLayer> deviceLayer_;
  rt::RuntimePtr runtime_;
  std::vector<rt::DeviceId> devices_;
  std::vector<rt::StreamId> defaultStreams_;
};

template <typename TContainer> void randomize(TContainer& container, int init, int end) {
  static std::mt19937 gen(std::random_device{}());
  static std::uniform_int_distribution dis(init, end);
  for (auto& v : container) {
    v = static_cast<typename TContainer::value_type>(dis(gen));
  }
}

float getRand() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<float> dis(0, 1.0f); // rage 0 - 1
  return dis(e);
}

// obtained from et-trace/encoder.h...
// need to review deps and conans with et-trace.
// constexpr uint32_t TRACE_EVENT_ENABLE_ALL = 0xFFFFFFFFU;
// constexpr uint32_t TRACE_FILTER_ENABLE_ALL = 0xFFFFFFFFU;

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

constexpr bool enableKernelTraces = true;

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

class Sample1 : public Test {
public:
  Sample1() = delete;
  Sample1(const Config& config)
    : Test(config) {
  }

  void run() override {
    auto kernel = loadKernel(kernelPath.getValue());
    auto numElems = 150U;
    auto hSrc1 = std::vector<int>(numElems);
    auto hSrc2 = std::vector<int>(numElems);
    auto hDst = std::vector<int>(numElems);
    auto dSrc1 = runtime_->mallocDevice(devices_[0], numElems * sizeof(int));
    auto dSrc2 = runtime_->mallocDevice(devices_[0], numElems * sizeof(int));
    auto dDst = runtime_->mallocDevice(devices_[0], numElems * sizeof(int));
    randomize(hSrc1, std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max());
    randomize(hSrc2, std::numeric_limits<int>::lowest(), std::numeric_limits<int>::max());

    struct Params {
      void* src1;
      void* src2;
      void* dst;
      int elements;
    } __attribute__((packed));

    Params params{dSrc1, dSrc2, dDst, static_cast<int>(numElems)};

    runtime_->memcpyHostToDevice(defaultStreams_[0], reinterpret_cast<std::byte*>(hSrc1.data()), dSrc1,
                                 numElems * sizeof(int));
    runtime_->memcpyHostToDevice(defaultStreams_[0], reinterpret_cast<std::byte*>(hSrc2.data()), dSrc2,
                                 numElems * sizeof(int));

    constexpr bool barrier = true;
    constexpr bool flushL3 = false;
    constexpr uint64_t shireMask = 0x1ffffffff;
    //    constexpr uint64_t shireMask = 0x1;

    // Alloc space on device to get user traces.
    auto traceDeviceBuffer = runtime_->mallocDevice(devices_[0], kTraceBufferSize);

    std::optional<rt::UserTrace> optUserTrace = getKernelTraceParams(traceDeviceBuffer, kTraceBufferSize);

    runtime_->kernelLaunch(defaultStreams_[0], kernel, reinterpret_cast<std::byte*>(&params), sizeof(params), shireMask,
                           barrier, flushL3, optUserTrace);

    runtime_->memcpyDeviceToHost(defaultStreams_[0], dDst, reinterpret_cast<std::byte*>(hDst.data()),
                                 numElems * sizeof(int));

    // geting device traces.
    std::vector<std::byte> deviceTrace(kTraceBufferSize);
    runtime_->memcpyDeviceToHost(defaultStreams_[0], traceDeviceBuffer, deviceTrace.data(), deviceTrace.size());

    runtime_->waitForStream(defaultStreams_[0]);

    // serialize traces to disck
    auto tracePath =
      std::filesystem::current_path() / std::filesystem::path("traceKernels_dev" + std::to_string(0) + ".bin");
    auto traceStream = std::ofstream(tracePath, std::ios::binary | std::ios::out);
    traceStream.write((char*)deviceTrace.data(), deviceTrace.size());

    // cleanups on device.
    runtime_->unloadCode(kernel);
    runtime_->freeDevice(devices_[0], dSrc1);
    runtime_->freeDevice(devices_[0], dSrc2);
    runtime_->freeDevice(devices_[0], dDst);
    runtime_->freeDevice(devices_[0], traceDeviceBuffer);

#if 0
    for (auto i = 0U; i < numElems; ++i) {
      assert(hDst[i] == (hSrc1[i] + hSrc2[i] + (i == 123 ? 1 : 0)));
    }
#endif
    std::cout << "DONE\n";
  }
};

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Basic ET-SoC1 host kernel launcher app\n\n");
  Config config;
  config.dump();
  Sample1 sample(config);
  sample.setup();
  sample.run();
  sample.tearDown();
  return 0;
}
