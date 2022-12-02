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

// Specific kernel lancuher class.
class HelloWorld : public GenericLauncher {
public:
  HelloWorld() = delete;
  HelloWorld(const Config& config)
    : GenericLauncher(config){};

  void prepareKernelArguments() override {
    kernelArgs_ = nullptr;
    kernelArgsSize_ = 0;
  }

};

DEFINE_string(device_type, "sysemu", "Device Type to be used (sysemu,fake,silicon)");
DEFINE_uint64(kernel_launch_timeout, 10, "timeout (inseconds) to wait for kernelLaunch");
DEFINE_uint64(num_launches, 1, "Number of times the kernel will be launched");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  Config config{modeFromString(FLAGS_device_type), 1};
  
  HelloWorld launcher(config);
  launcher.initialize();       // loadCode() happens in initialize.
  launcher.prepareKernelArguments();

  auto timeout = std::chrono::seconds(FLAGS_kernel_launch_timeout);
  for (size_t i = 0; i < FLAGS_num_launches; i++) {
    launcher.kernelLaunch();
    launcher.waitKernelCompletion(timeout);
    launcher.dumpTracesToFile(i);
  }

  launcher.deInitialize();
  launcher.tearDown();

  return 0;
}

