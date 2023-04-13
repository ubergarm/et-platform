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
#include <iterator>
#include <runtime/DeviceLayerFake.h>
#include <sw-sysemu/SysEmuOptions.h>

#include <getopt.h>

#include "GenericLauncher.h"
#include "RuntimeImpWithCoreDump.h"

// Trace Buffer realted constants.
constexpr size_t kTraceBytesPerHart = 4096;
constexpr size_t kNumHarts = 2048;
constexpr size_t kTraceBufferSize = kTraceBytesPerHart * kNumHarts;
constexpr bool enableKernelTraces = true;

emu::SysEmuOptions getDefaultOptions(std::string const &simulator_params) {

  constexpr uint64_t kSysEmuMaxCycles = std::numeric_limits<uint64_t>::max();
  constexpr uint64_t kSysEmuMinionShiresMask = 0x1FFFFFFFFu;

  emu::SysEmuOptions sysEmuOptions;

  sysEmuOptions.runDir = std::filesystem::current_path();
  sysEmuOptions.maxCycles = kSysEmuMaxCycles;
  sysEmuOptions.minionShiresMask = kSysEmuMinionShiresMask;
  sysEmuOptions.puUart0Path = sysEmuOptions.runDir + "/pu_uart0_tx.log";
  sysEmuOptions.puUart1Path = sysEmuOptions.runDir + "/pu_uart1_tx.log";
  sysEmuOptions.spUart0Path = sysEmuOptions.runDir + "/spio_uart0_tx.log";
  sysEmuOptions.spUart1Path = sysEmuOptions.runDir + "/spio_uart1_tx.log";
  sysEmuOptions.startGdb = false;

  // Pass the sysemu parameters from command line
  auto cmd = simulator_params;
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
    std::cout << "Running tests with PCIE deviceLayer\n";
    deviceLayer_ = dev::IDeviceLayer::createPcieDeviceLayer();
    break;
  case Mode::SYSEMU: {
    std::cout << "Running tests with SYSEMU deviceLayer\n";
    auto opts = getDefaultOptions(simulator_params_);
    std::vector<decltype(opts)> vopts;
    for (auto i = 0; i < config_.numDevices_; ++i) {
      vopts.emplace_back(opts);
      vopts.back().logFile += std::to_string(i);
    }
    deviceLayer_ = dev::IDeviceLayer::createSysEmuDeviceLayer(vopts);
    break;
  }
  case Mode::FAKE:
    std::cout << "Running tests with FAKE deviceLayer\n";
    deviceLayer_ = std::make_unique<dev::DeviceLayerFake>();
    options.checkDeviceApiVersion_ = false;
    break;
  case Mode::LAST:
    std::cout << "Unsupported device \n";
    exit(-1);
    break;
  }

  createRuntime(enableCoreDump_, options);

  devices_ = runtime_->getDevices();

  for (auto i = 0U; i < static_cast<uint32_t>(deviceLayer_->getDevicesCount()); ++i) {
    defaultStreams_.emplace_back(runtime_->createStream(devices_[i]));
    traceStreams_.emplace_back(runtime_->createStream(devices_[i]));
    numDev_++;
  }

  // Program callbacks for error management.
  auto streamErrorHandler = [this, rt = getRuntime(enableCoreDump_)]([[maybe_unused]] rt::EventId id,
                                                                     const rt::StreamError& error) {
    std::cout << "streamErrorHandler on deviceId[" << std::to_string((int)error.device_) << "] "
              << "() rt reports an error on a stream command(EventId: " << static_cast<int>(id) << "):\n"
              << error.getString();
    if ((error.errorCode_ == rt::DeviceErrorCode::DmaHostAborted) or
        (error.errorCode_ == rt::DeviceErrorCode::KernelLaunchHostAborted)) {
      std::cout << std::to_string(error.errorCode_) << " Errors during aborts are expected, ignoring\n";
      return;
    }

    kernelError_++;

    if (enableCoreDump_) {
      abortManager_.dumpCore(rt, id, error);
    }
  };

  // Program callback when we want kernel aborts (due to a timeout) to dump corefiles
  auto abortedKernelHandler = [this, rt = getRuntime(enableCoreDump_)](rt::EventId id, std::byte const* context,
                                                                       size_t size,
                                                                       std::function<void()> freeResources) {
    std::cout << "abortedKernelHandler"
              << " () rt reports that a kernel has been aborted (EventId: " << static_cast<int>(id) << ")\n";
    kernelAbort_++;

    if (enableCoreDump_) {
      auto error = abortManager_.handleAbortedKernelAndDumpCore(rt, id, context, size, freeResources);
      if (error.has_value()) {
        reportUserException(error.value());
      }
    }
  };

  runtime_->setOnStreamErrorsCallback(streamErrorHandler);
  runtime_->setOnKernelAbortedErrorCallback(abortedKernelHandler);

  // Alloc space on device for user traces. Note: This buffer will be reused across differnet kernel launches.
  if (enableKernelTraces) {
    for (uint32_t idx = 0; idx < numDev_; idx++) {
      traceDeviceBuffer_.emplace_back(runtime_->mallocDevice(devices_[idx], kTraceBufferSize));
    }
  }
}

void GenericLauncher::unLoadKernel(rt::KernelId kernelId) {
  runtime_->unloadCode(kernelId);
}

void GenericLauncher::tearDown() {

  if(enableKernelTraces) {
    for (uint32_t deviceIdx = 0; deviceIdx < numDev_; deviceIdx++) {
      runtime_->freeDevice(devices_[deviceIdx], traceDeviceBuffer_[deviceIdx]);
    }
  }

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

  resetRuntime(enableCoreDump_);
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
  traceParams.threshold_ = 0;
  return traceParams;
}

void GenericLauncher::dumpTracesToFile(uint64_t fileIdx, rt::KernelId kernelId, uint32_t deviceIdx) {
  if (not enableKernelTraces) {
    return;
  }
  // geting device traces.
  std::vector<std::byte> deviceTrace(kTraceBufferSize);
  runtime_->memcpyDeviceToHost(traceStreams_[deviceIdx], traceDeviceBuffer_[deviceIdx], deviceTrace.data(),
                               deviceTrace.size());
  auto tracesTimeout = std::chrono::seconds(10);
  auto success = runtime_->waitForStream(traceStreams_[deviceIdx], tracesTimeout);

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
    std::filesystem::current_path() / std::filesystem::path("traceKernels_dev" + std::to_string(devIdx_) + "_" +
                                                            std::to_string(fileIdx) + traceSuffix + ".bin");
  auto traceStream = std::ofstream(tracePath, std::ios::binary | std::ios::out);
  traceStream.write((char*)deviceTrace.data(), deviceTrace.size());
}

void GenericLauncher::waitKernelCompletion(std::chrono::seconds timeout, uint32_t deviceIdx) {
  auto success = runtime_->waitForStream(defaultStreams_[deviceIdx], timeout);

  if (success) {
    return;
  }
  // Kernel did not complete on the expected time. let's abort the stream in which
  // the kernel is running.
  std::cout << "[TIMEOUT] " << __func__ << "() Wait for Stream command exceeded " << std::dec << int(timeout.count())
            << " seconds.  Aborting stream\n";

  auto event = runtime_->abortStream(defaultStreams_[deviceIdx]);
  auto abortTimeout = std::chrono::seconds(10);
  success = runtime_->waitForEvent(event, abortTimeout);
  if (success) {
    std::cout << "[TIMEOUT] " << __func__ << "() event completed succesfuly: " << (int)event << "\n";
    return;
  }
  std::cout << "[TIMEOUT] " << __func__
             << "() timeout expired wating for abortStream event: "
             << (int)event << " to complete\n";
  return;
}

void GenericLauncher::doKernelLaunch(rt::KernelId kernelId, std::byte* params, size_t size, uint64_t shireMask,
                                     uint32_t deviceIdx) {
  std::optional<rt::UserTrace> optUserTrace = fillKernelTraceParams(traceDeviceBuffer_[deviceIdx], kTraceBufferSize);
  constexpr bool barrier = true;
  constexpr bool flushL3 = false;

  runtime_->kernelLaunch(defaultStreams_[deviceIdx], kernelId, params, size, shireMask, barrier, flushL3, optUserTrace);
}

void GenericLauncher::reportUserException(const rt::StreamError& error) const {
  std::cout << "Exception found, need to dump the execution context\n";
  // Dump execution context into a file
  auto path = std::experimental::filesystem::current_path();
  auto filename = "device_"+std::to_string((int)error.device_)+"_execution_context.txt";
  std::ofstream out(path / filename);
  out << error.getString();
  out << "\n---\n";
  out.close();
}

void GenericLauncher::createRuntime(bool enableCoreDump, rt::Options options) {
  if (enableCoreDump) {
    // If core dump is enabled, runtime wrapper with core dump capabilities is used
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

// Passes pointer to runtime instance without core dump capabilities
// to abortedKernelHandler callback.
// Runtime is used inside the callback to copy error context
// and dump core from device to host.
rt::IRuntime* GenericLauncher::getRuntime(bool enableCoreDump) {
  if (enableCoreDump) {
    return runtimeBase_.get();
  } else {
    return runtime_.get();
  }
}

void GenericLauncher::parse_args(int argc, char* const* argv) {

  static constexpr const char* short_opts = "cs:";

  static const std::vector<struct option> long_opts_vect {{"enableCoreDump", no_argument, nullptr, 'c'},
                                                          {"simulator_params", required_argument, nullptr, 's'},
                                                          {nullptr, 0, nullptr, 0}};

  int ret = 0;
  int index = 0;
  opterr = 0;

  /*
    A program that scans multiple argument vectors, or rescans the same vector more than once,
    and wants to make use of GNU extensions such as '+' and '-' at the start of optstring,
    or changes the value of POSIXLY_CORRECT between scans, must reinitialize getopt() by
    resetting optind to 0, rather than the traditional value of 1. (Resetting to 0 forces
    the invocation of an internal initialization routine that rechecks POSIXLY_CORRECT
    and checks for GNU extensions in optstring.)
  */

  optind = 0;

  while ((ret = getopt_long(argc, argv, short_opts, long_opts_vect.data(), &index)) != -1) {
    switch (ret) {
    case 'c':
      enableCoreDump_ = true;
      break;
    case 's':
      simulator_params_ = optarg;
      break;
    default:
      break;
    }
  }

  /* It needs to do again because on invoke sysemu if is the case, It calls getopts again */
  optind = 0;
}
