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
#include <iostream>
#include <numeric>

#include "saxpy_kernel_arguments.h"

// Host-side implementation of a SAXPY kernel (for checking purposes)
template <class InputIt, class OutputIt, class VT = typename InputIt::value_type>
OutputIt saxpy(InputIt first, InputIt last, OutputIt d_first, VT a) {
  for (; first != last; ++first) {
    *d_first = a * *first + *d_first;
    ++d_first;
  }
  return d_first;
}

// Specific kernel launcher class.
class Saxpy : public GenericLauncher {
public:
  Saxpy() = delete;
  Saxpy(const Config& config)
    : GenericLauncher(config){};

  void prepareInput() {
    a_ = 3;
    std::iota(x_.begin(), x_.end(), 0);
    std::iota(y_.begin(), y_.end(), 100);
  }

  void performDeviceAllocs() {
    deviceX_ = runtime_->mallocDevice(devices_[devIdx_], x_.size() * sizeof(float));
    deviceY_ = runtime_->mallocDevice(devices_[devIdx_], y_.size() * sizeof(float));
  }

  void programHost2DevCopies() {
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte*)x_.data(), deviceX_, x_.size() * sizeof(float));
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte*)y_.data(), deviceY_, y_.size() * sizeof(float));
  }

  void programDev2HostCopies() {
    runtime_->memcpyDeviceToHost(defaultStreams_[devIdx_], deviceY_, (std::byte*)y_.data(), y_.size() * sizeof(float));
  }

  void freeDeviceAllocs() {
    runtime_->freeDevice(devices_[devIdx_], deviceX_);
    runtime_->freeDevice(devices_[devIdx_], deviceY_);
  }

  static constexpr size_t numElems_ = 256;
  float a_;
  std::vector<float> x_ = std::vector<float>(numElems_);
  std::vector<float> y_ = std::vector<float>(numElems_);
  std::byte* deviceX_;
  std::byte* deviceY_;
};

DEFINE_string(device_type, "sysemu", "Device type to be used (sysemu,fake,silicon)");
DEFINE_uint64(kernel_launch_timeout, 10, "Timeout (in seconds) to wait for kernelLaunch");
DEFINE_uint64(num_launches, 1, "Number of times the kernel will be launched");
DEFINE_string(kernel_path, "", "ET-SoC-1 kernel path and filename");
DEFINE_uint64(launch_mult, 1, "Number of times the kernel is executed for each launch");
DEFINE_double(epsilon, 0.0, "Delta used for comparison between host and device");

int main(int argc, char** argv) {

  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  Config config{modeFromString(FLAGS_device_type), 1};
  config.dump();

  Saxpy launcher(config);
  launcher.initialize();
  auto kernelId = launcher.loadKernel(FLAGS_kernel_path);
  launcher.performDeviceAllocs();
  launcher.prepareInput();

  // Copy original values to check them later
  std::vector<float> x2 = launcher.x_;
  std::vector<float> y2 = launcher.y_;

  for (size_t i = 0; i < FLAGS_num_launches; i++) {
    launcher.programHost2DevCopies();

    KernelArguments kernelArgs{launcher.x_.size(), (float*)launcher.deviceX_, (float*)launcher.deviceY_, launcher.a_};

    launcher.kernelLaunch(kernelId, &kernelArgs);
    launcher.programDev2HostCopies();
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

  // Check kernel results
  for (size_t i = 0; i < FLAGS_num_launches * FLAGS_launch_mult; ++i) {
    saxpy(x2.begin(), x2.end(), y2.begin(), launcher.a_);
  }
  if (!std::equal(y2.begin(), y2.end(), launcher.y_.begin(),
                  [=](float host, float dev) { return std::abs(host - dev) <= FLAGS_epsilon; })) {
    std::cerr << "error: SAXPY host/device results do not match" << std::endl;
    return 1;
  }

  return 0;
}
