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

// TODO
// FIXME: In case it is needed we should have or own gp-sdk logger,
// although it only affects runtime.
#include <hostUtils/logging/Logger.h>

#include <string>
#include <vector>

enum class Mode { PCIE, SYSEMU, FAKE, LAST };

static inline std::string to_string(Mode m) {
  const std::vector<std::string> modes = {"Hardware", "Emulator", "Fake"};
  return m >= Mode::LAST ? "unknown" : modes[int(m)];
}

class Config {
public:
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

class GenericLauncher {
public:
  GenericLauncher() = delete;
  GenericLauncher(const Config& config)
    : config_(config){};
  void initialize(); // setup
  void deInitialize();
  void tearDown(); // FIXME:consolidate with deinitialize for multi-inference use-cases
  void dumpTracesToFile();

  void kernelLaunch();
  void waitKernelCompletion(std::chrono::seconds timeout);

  virtual void prepareKernelArguments() = 0;

  inline static Mode mode_ = Mode::SYSEMU;
  inline static uint32_t numDevices_ = 1;

private:
  rt::KernelId loadKernel(const std::string& kernelName, uint32_t deviceIdx = 0);
  std::vector<std::byte> readFile(const std::string& path);
  // FIXME: just to enable glog-logger on runtime.
  logging::LoggerDefault loggerDefault_;

protected:
  const Config& config_;
  std::unique_ptr<dev::IDeviceLayer> deviceLayer_;
  rt::RuntimePtr runtime_;
  std::vector<rt::DeviceId> devices_;
  std::vector<rt::StreamId> defaultStreams_;
  std::vector<rt::StreamId> traceStreams_;
  
  std::byte* traceDeviceBuffer_;
  // todo: support loading multiple kernels and launch named kernels.
  rt::KernelId kernel_;
  std::byte* kernelArgs_;
  size_t kernelArgsSize_;
  // TODO: multi-dev design.
  size_t devIdx_ = 0;
};

#endif // GENERIC_LAUNCHER_H
