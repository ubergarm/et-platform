//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------
#include "GenericLauncher.h"
#include <gflags/gflags.h>
#include <numeric>

#include "barrierKernelArguments.h"

// Specific kernel lancuher class.
class BarrierLauncher : public GenericLauncher {
public:
  BarrierLauncher() = delete;
  BarrierLauncher(const Config& config)
    : GenericLauncher(config){};

  void performDeviceAllocs() {
    deviceData_ = runtime_->mallocDevice(devices_[devIdx_], data_.size() * sizeof(uint64_t));
    deviceAccumData_ = runtime_->mallocDevice(devices_[devIdx_], accumData_.size() * sizeof(uint64_t));
  }

  void programHost2DevCopies() {
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte *) data_.data(), deviceData_,
                                 data_.size() * sizeof(uint64_t));
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte *)  accumData_.data(), deviceAccumData_,
                                 accumData_.size() * sizeof(uint64_t));
  }

  void freeDeviceAllocs() {
    runtime_->freeDevice(devices_[devIdx_], deviceData_);
    runtime_->freeDevice(devices_[devIdx_], deviceAccumData_);
  }

  static constexpr size_t assignedMinions = 32;
  std::vector<uint64_t> data_ = std::vector<uint64_t>(assignedMinions, 0);
  std::vector<uint64_t> accumData_ = std::vector<uint64_t>(assignedMinions, 0);
  std::byte* deviceData_;
  std::byte* deviceAccumData_;
};

DEFINE_string(device_type, "sysemu", "Device Type to be used (sysemu,fake,silicon)");
DEFINE_uint64(kernel_launch_timeout, 10, "timeout (inseconds) to wait for kernelLaunch");
DEFINE_uint64(num_launches, 1, "Number of times the kernel will be launched");
DEFINE_string(kernel_path, "", "ET-SoC-1 kernel path and filename");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  Config config{modeFromString(FLAGS_device_type), 1};
  config.dump();

  BarrierLauncher launcher(config);
  launcher.initialize();
  auto kernelId = launcher.loadKernel(FLAGS_kernel_path);
  launcher.performDeviceAllocs();

  KernelArguments kernelArgs {launcher.data_.size(), (uint64_t *) launcher.deviceData_,
                        (uint64_t *) launcher.deviceAccumData_};

  for (size_t i = 0; i < FLAGS_num_launches; i++) {
    launcher.programHost2DevCopies();
    launcher.kernelLaunch(kernelId, &kernelArgs);
    // launcher.programDev2HostCopies();
    auto timeout = std::chrono::seconds(FLAGS_kernel_launch_timeout);
    launcher.waitKernelCompletion(timeout);
    launcher.dumpTracesToFile(i);

    if (launcher.kernelError_ || launcher.kernelAbort_) {
      return -1;
    }
  }

  launcher.freeDeviceAllocs();
  launcher.unLoadKernel(kernelId); 
  launcher.tearDown();

  return 0;
}
