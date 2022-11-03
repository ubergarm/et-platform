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
struct Params {
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

  void prepareKernelArguments() override {
    params_.src1 = dSrc1_;
    params_.src2 = dSrc2_;
    params_.dst = dDst_;
    params_.elements = int(hSrc1_.size());

    kernelArgs_ = (std::byte*)&params_;
    kernelArgsSize_ = sizeof(params_);
  }

private:
  static constexpr size_t numElems_ = 150;
  std::vector<int> hSrc1_ = std::vector<int>(numElems_);
  std::vector<int> hSrc2_ = std::vector<int>(numElems_);
  std::vector<int> hDst_ = std::vector<int>(numElems_);
  std::byte* dSrc1_;
  std::byte* dSrc2_;
  std::byte* dDst_;
  Params params_;
};

DEFINE_string(device_type, "sysemu", "Device Type to be used (sysemu,fake,silicon)");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  Config config{modeFromString(FLAGS_device_type), 1};
  config.dump();

  Launcher launcher(config);
  launcher.initialize();
  launcher.performDeviceAllocs();
  launcher.programHost2DevCopies();
  launcher.prepareKernelArguments();
  launcher.kernelLaunch();
  launcher.programDev2HostCopies();

  auto timeout = std::chrono::seconds(10);
  launcher.waitKernelCompletion(timeout);

  launcher.dumpTracesToFile();
  launcher.freeDeviceAllocs();
  launcher.deInitialize();
  launcher.tearDown();

  return 0;
}
