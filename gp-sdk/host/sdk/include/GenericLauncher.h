//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------
#ifndef GENERIC_LAUNCHER_H
#define GENERIC_LAUNCHER_H
#include <device-layer/IDeviceLayer.h>
#include <runtime/IRuntime.h>

#include <hostUtils/logging/Logger.h>

#include <string>
#include <vector>

enum class Mode { PCIE, SYSEMU, FAKE, LAST };

static inline std::string to_string(Mode m) {
  const std::vector<std::string> modes = {"silicon", "sysemu", "fake"};
  return m >= Mode::LAST ? "unknown" : modes[int(m)];
}

static inline Mode modeFromString(std::string m) {
  std::map<std::string, Mode> modeFromStr = {{"sysemu", Mode::SYSEMU}, {"fake", Mode::FAKE}, {"silicon", Mode::PCIE}};
  return modeFromStr.count(m) ? modeFromStr[m] : Mode::LAST;
}

struct Config {
  void dump() const {
    std::cout << "Selected options:\n";
    std::cout << "  Device type: " << to_string(mode_) << "\n";
    std::cout << "  Device count: " << numDevices_ << "\n";
  }

  Mode mode_{Mode::SYSEMU};
  size_t numDevices_{1};
};

class GenericLauncher {
public:
  GenericLauncher() = delete;
  GenericLauncher(const Config& config)
    : config_(config){};
  void initialize(); // setup
  void tearDown(); 
  void dumpTracesToFile(uint64_t fileIdx = 0, rt::KernelId kernelId = (rt::KernelId)(-1));

  template <typename TParams>
  void kernelLaunch(rt::KernelId kernelId, TParams * params, uint64_t shireMask = 0xffffffff) {
    doKernelLaunch(kernelId, (std::byte *)params, sizeof(TParams), shireMask);
  }

  void kernelLaunch(rt::KernelId kernelId, uint64_t shireMask = 0xffffffff) {
    doKernelLaunch(kernelId, nullptr, 0, shireMask);
  }
  
  void waitKernelCompletion(std::chrono::seconds timeout);
  rt::KernelId loadKernel(const std::string& kernelName, uint32_t deviceIdx = 0);
  void unLoadKernel(rt::KernelId kernelId);
  
  std::atomic<uint64_t> kernelError_ = 0; // Number of kernels that reported an error
  std::atomic<uint64_t> kernelAbort_ = 0; // Number of kernels aborted

private:
  std::vector<std::byte> readFile(const std::string& path);
  // just to enable glog-logger on runtime.
  logging::LoggerDefault loggerDefault_;
  void doKernelLaunch(rt::KernelId, std::byte * params, size_t size, uint64_t shireMask);
  
protected:
  const Config& config_;
  std::unique_ptr<dev::IDeviceLayer> deviceLayer_;
  rt::RuntimePtr runtime_;
  std::vector<rt::DeviceId> devices_;
  std::vector<rt::StreamId> defaultStreams_;
  std::vector<rt::StreamId> traceStreams_;

  std::byte* traceDeviceBuffer_;
  // TODO: multi-dev design.
  size_t devIdx_ = 0;
};

#endif // GENERIC_LAUNCHER_H
