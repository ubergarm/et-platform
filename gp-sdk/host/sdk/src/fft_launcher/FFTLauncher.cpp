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

// Shared kernel-arguments with device:
// FIXME: move to a public header.
// beware of void * and use fixed-with instead.

struct TensorDesc {
  uint64_t nDims;
  uint64_t dims[6] = {0};
  uint64_t strides[6] = {0};
  uint64_t deviceAddress;
} __attribute__((packed));

struct FFTKernelArgs {
  uint32_t nTensors;
  TensorDesc tensors[2];
  uint32_t operation;
} __attribute__((packed));

enum class FFTOp { FFT = 0, IFFT = 1, SKIP = 1024 };

constexpr size_t heigh = 256;
constexpr size_t width = 256;
constexpr size_t planes = 2;
constexpr size_t channels = 3;
constexpr size_t batch = 1;

// dnnlib FFT kernel lancuher class.
class FFTLauncher : public GenericLauncher {
public:
  FFTLauncher() = delete;
  FFTLauncher(const Config& config)
    : GenericLauncher(config){};

  void prepareInput() {
    //  TODO: initialize with meaningful data.
  }

  void performDeviceAllocs() {
    devInputTensor = runtime_->mallocDevice(devices_[devIdx_], inputTensor.size() * sizeof(float));
    devOutputTensor = runtime_->mallocDevice(devices_[devIdx_], outputTensor.size() * sizeof(float));
  }

  void programHost2DevCopies() {
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte*)inputTensor.data(), devInputTensor,
                                 inputTensor.size() * sizeof(float));
  }

  void programDev2HostCopies() {
    runtime_->memcpyDeviceToHost(defaultStreams_[devIdx_], devOutputTensor, (std::byte*)outputTensor.data(),
                                 outputTensor.size() * sizeof(float));
  }

  void freeDeviceAllocs() {
    runtime_->freeDevice(devices_[devIdx_], devInputTensor);
    runtime_->freeDevice(devices_[devIdx_], devOutputTensor);
  }

  void prepareKernelArguments() override {
    // TODO: implement a nicer TensorDesc constructor...but keep it POD format.
    TensorDesc inputTensorDesc = {5,
                                  {batch, channels, planes, heigh, width},
                                  {width * heigh * planes * channels, width * heigh * planes, width * heigh, width, 1},
                                  uint64_t(devInputTensor)};
    TensorDesc outputTensorDesc = {5,
                                   {batch, channels, planes, heigh, width},
                                   {width * heigh * planes * channels, width * heigh * planes, width * heigh, width, 1},
                                   uint64_t(devOutputTensor)};

    fftKernelArgs_.nTensors = 2;
    fftKernelArgs_.tensors[0] = inputTensorDesc;
    fftKernelArgs_.tensors[1] = outputTensorDesc;
    fftKernelArgs_.operation = uint64_t(FFTOp::FFT);

    kernelArgs_ = (std::byte*)&fftKernelArgs_;
    kernelArgsSize_ = sizeof(fftKernelArgs_);
  }

private:
  std::vector<float> inputTensor = std::vector<float>(batch * channels * planes * width * heigh);
  std::vector<float> outputTensor = std::vector<float>(batch * channels * planes * width * heigh);
  std::byte* devInputTensor = nullptr;
  std::byte* devOutputTensor = nullptr;
  FFTKernelArgs fftKernelArgs_;
};

DEFINE_string(device_type, "sysemu", "Device Type to be used (sysemu,fake,silicon)");
DEFINE_uint64(kernel_launch_timeout, 10, "timeout (inseconds) to wait for kernelLaunch");

int main(int argc, char** argv) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  Config config{modeFromString(FLAGS_device_type), 1};
  config.dump();

  FFTLauncher launcher(config);
  launcher.initialize();
  launcher.performDeviceAllocs();
  launcher.programHost2DevCopies();
  launcher.prepareKernelArguments();

  launcher.prepareInput();
  launcher.kernelLaunch();
  launcher.programDev2HostCopies();

  auto timeout = std::chrono::seconds(FLAGS_kernel_launch_timeout);
  launcher.waitKernelCompletion(timeout);

  launcher.dumpTracesToFile();
  launcher.freeDeviceAllocs();
  launcher.deInitialize();
  launcher.tearDown();

  return 0;
}
