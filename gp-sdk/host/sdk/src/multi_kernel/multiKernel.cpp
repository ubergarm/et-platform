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


#include "saxpy_kernel_arguments.h"

// Specific kernel lancuher class.
class MultiKernel : public GenericLauncher {
public:
  MultiKernel() = delete;
  MultiKernel(const Config& config)
    : GenericLauncher(config){};

  void prepareInputSapxy() {
    a_= 3;
    std::iota(x_.begin(), x_.end() ,0);
    std::iota(y_.begin(), y_.end() ,100);

  }

  void performDeviceAllocsSaxpy() {
    deviceX_ = runtime_->mallocDevice(devices_[devIdx_], x_.size() * sizeof(float));
    deviceY_ = runtime_->mallocDevice(devices_[devIdx_], y_.size() * sizeof(float));
  }

  void programHost2DevCopiesSaxpy() {
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte *) x_.data(), deviceX_,
                                 x_.size() * sizeof(float));
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte *)  y_.data(), deviceY_,
                                 y_.size() * sizeof(float));
  }

  void programDev2HostCopiesSaxpy() {
    runtime_->memcpyDeviceToHost(defaultStreams_[devIdx_],deviceY_, (std::byte *)  y_.data(),
                                 y_.size() * sizeof(float));
  }

  void freeDeviceAllocsSaxpy() {
    runtime_->freeDevice(devices_[devIdx_], deviceX_);
    runtime_->freeDevice(devices_[devIdx_], deviceY_);
  }


  static constexpr size_t numElems_ = 256;
  float a_;
  std::vector<float> x_ = std::vector<float>(numElems_);
  std::vector<float> y_ = std::vector<float>(numElems_);
 
  std::byte* deviceX_;
  std::byte* deviceY_;
  std::vector<rt::KernelId> kernels_;
};


DEFINE_string(device_type, "sysemu", "Device Type to be used (sysemu,fake,silicon)");
DEFINE_uint64(kernel_launch_timeout, 10, "timeout (inseconds) to wait for kernelLaunch");
DEFINE_uint64(num_launches, 1, "Number of times the kernel will be launched");
DEFINE_string(kernel_path1, "", "ET-SoC-1 kernel first path and filename (no data context kernel (just test/tracing e.g hello-world) ");
DEFINE_string(kernel_path2, "", "ET-SoC-1 kernel second path and filename (saxpy kernel)");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  Config config{modeFromString(FLAGS_device_type), 1};
  config.dump();

  
  MultiKernel launcher(config);
  launcher.initialize();

  auto kernelId = launcher.loadKernel(FLAGS_kernel_path1);
  launcher.kernels_.emplace_back(kernelId);
  kernelId = launcher.loadKernel(FLAGS_kernel_path2);
  launcher.kernels_.emplace_back(kernelId);

  std::cout<< "loadKernel --> " << FLAGS_kernel_path1 << " with kernel_id="<< int(launcher.kernels_[0]) << std::endl;    
  std::cout<< "loadKernel --> " << FLAGS_kernel_path2 << " with kernel_id="<< int(launcher.kernels_[1]) << std::endl;    
  
  auto timeout = std::chrono::seconds(FLAGS_kernel_launch_timeout);  
 

  // Assuming first kernel requires no data context (e.g hello-world and the like).
  launcher.kernelLaunch(launcher.kernels_[0]);
  launcher.waitKernelCompletion(timeout);
  launcher.dumpTracesToFile(0, launcher.kernels_[0]);

  if (launcher.kernelError_ || launcher.kernelAbort_) {
    std::cout<< "Error on kernel id="<< int(launcher.kernels_[0]) << std::endl;	
    return -1;
  }


  //assume saxpy as second elf to be loaded & prep data context for it. 
  launcher.performDeviceAllocsSaxpy();
  launcher.prepareInputSapxy();                   
  launcher.programHost2DevCopiesSaxpy();
  KernelArguments kernelArgs {launcher.x_.size(), (float *) launcher.deviceX_ ,  (float *) launcher.deviceY_, launcher.a_};

  launcher.kernelLaunch(launcher.kernels_[1], &kernelArgs);
  launcher.programDev2HostCopiesSaxpy();
  launcher.waitKernelCompletion(timeout);
  launcher.dumpTracesToFile(0, launcher.kernels_[1]);

  if (launcher.kernelError_ || launcher.kernelAbort_) {
    std::cout<< "Error on kernel id="<< int(launcher.kernels_[1]) << std::endl;	
    return -1;
  }
  
  launcher.freeDeviceAllocsSaxpy();

  launcher.unLoadKernel(launcher.kernels_[0]);  
  launcher.unLoadKernel(launcher.kernels_[1]);
  launcher.tearDown();
  
  return 0;
}
