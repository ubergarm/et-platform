//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#include "Launcher.h"
#include "llvm/Support/CommandLine.h"

void Launcher::performDeviceAllocs() {
  dSrc1_ = runtime_->mallocDevice(devices_[devIdx_], hSrc1_.size() * sizeof(int));
  dSrc2_ = runtime_->mallocDevice(devices_[devIdx_], hSrc2_.size() * sizeof(int));
  dDst_ = runtime_->mallocDevice(devices_[devIdx_], hDst_.size() * sizeof(int));
}

void Launcher::programHost2DevCopies() {
  runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], reinterpret_cast<std::byte*>(hSrc1_.data()), dSrc1_,
                               hSrc1_.size() * sizeof(int));
  runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], reinterpret_cast<std::byte*>(hSrc2_.data()), dSrc2_,
                               hSrc2_.size() * sizeof(int));
}

void Launcher::programDev2HostCopies() {
  runtime_->memcpyDeviceToHost(defaultStreams_[devIdx_], dDst_, reinterpret_cast<std::byte*>(hDst_.data()),
                               hDst_.size() * sizeof(int));
}
void Launcher::freeDeviceAllocs() {
  runtime_->freeDevice(devices_[devIdx_], dSrc1_);
  runtime_->freeDevice(devices_[devIdx_], dSrc2_);
  runtime_->freeDevice(devices_[devIdx_], dDst_);
}

void Launcher::prepareKernelArguments() {
  params_.src1 = dSrc1_;
  params_.src2 = dSrc2_;
  params_.dst = dDst_;
  params_.elements = int(hSrc1_.size());

  kernelArgs_ = (std::byte*)&params_;
  kernelArgsSize_ = sizeof(params_);
}

int main(int argc, char** argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Basic ET-SoC1 host kernel launcher app\n\n");
  Config config;
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
