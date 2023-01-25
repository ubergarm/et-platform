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
#include <cassert>
#include <gflags/gflags.h>
#include <numeric>

// Specific kernel lancuher class.
class MultiDevice : public GenericLauncher {
public:
  MultiDevice() = delete;
  MultiDevice(const Config& config)
    : GenericLauncher(config){};

  std::vector<rt::KernelId> kernels_;
};

DEFINE_string(device_type, "sysemu", "Device Type to be used (sysemu,fake,silicon)");
DEFINE_uint64(kernel_launch_timeout, 10, "timeout (inseconds) to wait for kernelLaunch");
DEFINE_uint64(num_launches, 1, "Number of times the kernel will be launched");
DEFINE_string(kernel_path1, "",
              "ET-SoC-1 kernel first path and filename (no data context kernel (just test/tracing e.g hello-world) ");
DEFINE_string(kernel_path2, "", "ET-SoC-1 kernel second path and filename (saxpy kernel)");
DEFINE_uint32(nDevices, 1, "Number of devices to be used.");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  Config config{modeFromString(FLAGS_device_type), FLAGS_nDevices};
  config.dump();

  MultiDevice launcher(config);
  launcher.initialize();

  if (modeFromString(FLAGS_device_type) == Mode::PCIE) {
    assert(launcher.getNumDevices() >= FLAGS_nDevices);
  }

  auto kernelId = launcher.loadKernel(FLAGS_kernel_path1, 0);
  launcher.kernels_.emplace_back(kernelId);
  kernelId = launcher.loadKernel(FLAGS_kernel_path2, 1);
  launcher.kernels_.emplace_back(kernelId);

  std::cout << "loadKernel --> " << FLAGS_kernel_path1 << " with kernel_id=" << int(launcher.kernels_[0]) << std::endl;
  std::cout << "loadKernel --> " << FLAGS_kernel_path2 << " with kernel_id=" << int(launcher.kernels_[1]) << std::endl;

  auto timeout = std::chrono::seconds(FLAGS_kernel_launch_timeout);

  // Launch a hang test over two devices kernels have to be hang.elf
  launcher.kernelLaunch(launcher.kernels_[0], (uint32_t)0);

  launcher.kernelLaunch(launcher.kernels_[1], (uint32_t)1);

  launcher.waitKernelCompletion(timeout, 0);
  launcher.dumpTracesToFile(0, launcher.kernels_[0], 0);

  launcher.waitKernelCompletion(timeout, 1);
  launcher.dumpTracesToFile(0, launcher.kernels_[1], 1);

  // kernelError_ and kernelAbort_ have to be catched asking to the specific device currently not allowed at runtime
  // side
  if (launcher.kernelError_ || launcher.kernelAbort_) {
    std::cout << "Error on kernel id=" << int(launcher.kernels_[0]) << "or in kernel id=" << int(launcher.kernels_[1])
              << std::endl;
    return -1;
  }

  launcher.unLoadKernel(launcher.kernels_[0]);
  launcher.unLoadKernel(launcher.kernels_[1]);
  launcher.tearDown();

  return 0;
}
