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


// Shared with device:
// FIXME: move to a public header.
// bewware of void * and use fixed-with instead.
struct KernelArguments {
  void* src1;
  void* src2;
  void* dst;
  int elements;
} __attribute__((packed));

// Specific kernel lancuher class.
class Launcher : public GenericLauncher {
public:
  Launcher() = delete;
  Launcher(const Config& config)
    : GenericLauncher(config){};

  void performDeviceAllocs() {
    dSrc1_ = runtime_->mallocDevice(devices_[devIdx_], hSrc1_.size() * sizeof(int));
    dSrc2_ = runtime_->mallocDevice(devices_[devIdx_], hSrc2_.size() * sizeof(int));
    dDst_ = runtime_->mallocDevice(devices_[devIdx_], hDst_.size() * sizeof(int));
  }

  void programHost2DevCopies() {
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], reinterpret_cast<std::byte*>(hSrc1_.data()), dSrc1_,
                                 hSrc1_.size() * sizeof(int));
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], reinterpret_cast<std::byte*>(hSrc2_.data()), dSrc2_,
                                 hSrc2_.size() * sizeof(int));
  }

  void programDev2HostCopies() {
    runtime_->memcpyDeviceToHost(defaultStreams_[devIdx_], dDst_, reinterpret_cast<std::byte*>(hDst_.data()),
                                 hDst_.size() * sizeof(int));
  }

  void freeDeviceAllocs() {
    runtime_->freeDevice(devices_[devIdx_], dSrc1_);
    runtime_->freeDevice(devices_[devIdx_], dSrc2_);
    runtime_->freeDevice(devices_[devIdx_], dDst_);
  }

  static constexpr size_t numElems_ = 150;
  std::vector<int> hSrc1_ = std::vector<int>(numElems_);
  std::vector<int> hSrc2_ = std::vector<int>(numElems_);
  std::vector<int> hDst_ = std::vector<int>(numElems_);
  std::byte* dSrc1_;
  std::byte* dSrc2_;
  std::byte* dDst_;

  KernelArguments args_ {(int *) dSrc1_, (int *) dSrc2_,
			 (int *) dDst_, int(hSrc1_.size())};
};

DEFINE_string(device_type, "sysemu", "Device Type to be used (sysemu,fake,silicon)");
DEFINE_uint64(kernel_launch_timeout, 10, "timeout (inseconds) to wait for kernelLaunch");
DEFINE_uint64(num_launches, 1, "Number of times the kernel will be launched");
DEFINE_string(kernel_path, "", "ET-SoC-1 kernel path and filename");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  Config config{modeFromString(FLAGS_device_type), 1};
  config.dump();

  Launcher launcher(config);
  launcher.initialize();
  auto kernel_id = launcher.loadKernel(FLAGS_kernel_path);
  launcher.kernels_.push_back(kernel_id);  
  launcher.performDeviceAllocs();

  for (size_t i = 0; i < FLAGS_num_launches; i++) {
    launcher.programHost2DevCopies();
    launcher.kernelLaunch(launcher.kernels_[0], &launcher.args_);
    launcher.programDev2HostCopies();
    auto timeout = std::chrono::seconds(FLAGS_kernel_launch_timeout);
    launcher.waitKernelCompletion(timeout);
    launcher.dumpTracesToFile(i);

    if (launcher.kernelError_ || launcher.kernelAbort_) {
      return -1;
    }
  }

  launcher.freeDeviceAllocs();
  launcher.unLoadKernel(launcher.kernels_[0]); 
  launcher.tearDown();

  return 0;
}
