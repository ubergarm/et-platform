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

static const std::string KERNELS_DIR = "/lib/esperanto-fw/kernels";

// Specific kernel lancuher class.
class MultiKernel : public GenericLauncher {
public:
  MultiKernel() = delete;
  MultiKernel(const Config& config)
    : GenericLauncher(config){};

  void prepareInput1() {
    a_= 3;
    std::iota(x_.begin(), x_.end() ,0);
    std::iota(y_.begin(), y_.end() ,100);

  }

  void performDeviceAllocs1() {
    deviceX_ = runtime_->mallocDevice(devices_[devIdx_], x_.size() * sizeof(float));
    deviceY_ = runtime_->mallocDevice(devices_[devIdx_], y_.size() * sizeof(float));
  }

  void programHost2DevCopies() {
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte *) x_.data(), deviceX_,
                                 x_.size() * sizeof(float));
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte *)  y_.data(), deviceY_,
                                 y_.size() * sizeof(float));
  }

  void programDev2HostCopies() {
    runtime_->memcpyDeviceToHost(defaultStreams_[devIdx_],deviceY_, (std::byte *)  y_.data(),
                                 y_.size() * sizeof(float));
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


DEFINE_string(device_type, "sysemu", "Device Type to be used (sysemu,fake,silicon)");
DEFINE_uint64(kernel_launch_timeout, 10, "timeout (inseconds) to wait for kernelLaunch");
DEFINE_uint64(num_launches, 1, "Number of times the kernel will be launched");
DEFINE_string(kernel_path1, "", "ET-SoC-1 kernel first path and filename");
DEFINE_string(kernel_path2, "", "ET-SoC-1 kernel second path and filename");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  Config config{modeFromString(FLAGS_device_type), 1};
  config.dump();

  
  MultiKernel launcher(config);
  launcher.initialize();

  rt::KernelId kernel_id = launcher.loadKernel(FLAGS_kernel_path1);
  auto timeout = std::chrono::seconds(FLAGS_kernel_launch_timeout);  
  std::cout<< "loadKernel --> " << FLAGS_kernel_path1 << " with kernel_id="<< int(kernel_id) << std::endl;    
  launcher.kernels_.push_back(kernel_id);
  
  launcher.kernelLaunch(launcher.kernels_[0]);
  launcher.waitKernelCompletion(timeout);
  launcher.dumpTracesToFile(0, launcher.kernels_[0]);

  if (launcher.kernelError_ || launcher.kernelAbort_) {
    std::cout<< "Error on launcher kernel id="<< int(launcher.kernels_[0]) << std::endl;	
    return -1;
  }

  kernel_id = launcher.loadKernel(FLAGS_kernel_path2); 
  std::cout<< "loadKernel --> " << FLAGS_kernel_path2 << " with kernel_id="<< int(kernel_id) << std::endl;    
  launcher.kernels_.push_back(kernel_id);

  //assume saxpy as second elf to be loaded 
  launcher.performDeviceAllocs1();
  launcher.prepareInput1();                   
  launcher.programHost2DevCopies();
  KernelArguments args1 {launcher.x_.size(), (float *) launcher.deviceX_ ,  (float *) launcher.deviceY_, launcher.a_};

  launcher.kernelLaunch(launcher.kernels_[1], &args1);
  launcher.programDev2HostCopies();
  launcher.waitKernelCompletion(timeout);
  launcher.dumpTracesToFile(0, launcher.kernels_[1]);

  if (launcher.kernelError_ || launcher.kernelAbort_) {
    std::cout<< "Error on launcher kernel id="<< int(launcher.kernels_[0]) << std::endl;	
    return -1;
  }
  
  launcher.freeDeviceAllocs();
  launcher.unLoadKernel(launcher.kernels_[0]);  
  launcher.unLoadKernel(launcher.kernels_[1]);
  launcher.tearDown();
  
  return 0;
}
