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

#include "txfma_kernel_arguments.h"

void txfma(float* C, const float* A, const float* B, size_t M, size_t K, size_t N) {
  std::fill_n(C, M * N, 0.f);
  for (auto k = 0; k < K; ++k) {
    for (auto m = 0; m < M; ++m) {
      for (auto n = 0; n < N; ++n) {
        C[m * N + n] += A[m * K + k] * B[k * N + n];
      }
    }
  }
}

// Kernel launcher for TxFMA
class TxFma : public GenericLauncher {
public:
  TxFma() = delete;
  TxFma(const Config& config)
    : GenericLauncher(config){};

  void prepareInput() {
    std::iota(A_.begin(), A_.end(), 0);
    std::iota(B_.begin(), B_.end(), 100);
  }

  void performDeviceAllocs() {
    deviceA_ = runtime_->mallocDevice(devices_[devIdx_], A_.size() * sizeof(float));
    deviceB_ = runtime_->mallocDevice(devices_[devIdx_], B_.size() * sizeof(float));
    deviceC_ = runtime_->mallocDevice(devices_[devIdx_], C_.size() * sizeof(float));
  }

  void programHost2DevCopies() {
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte*)A_.data(), deviceA_, A_.size() * sizeof(float));
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte*)B_.data(), deviceB_, B_.size() * sizeof(float));
  }

  void programDev2HostCopies() {
    runtime_->memcpyDeviceToHost(defaultStreams_[devIdx_], deviceC_, (std::byte*)C_.data(), C_.size() * sizeof(float));
  }

  void freeDeviceAllocs() {
    runtime_->freeDevice(devices_[devIdx_], deviceA_);
    runtime_->freeDevice(devices_[devIdx_], deviceB_);
    runtime_->freeDevice(devices_[devIdx_], deviceC_);
  }

  static constexpr uint32_t aRows = 14;
  static constexpr uint32_t aCols = 16;

  static constexpr uint32_t bRows = 16;
  static constexpr uint32_t bCols = 16;

  static constexpr uint32_t cRows = 14;
  static constexpr uint32_t cCols = 16;

  std::vector<float> A_ = std::vector<float>(aRows * aCols);
  std::vector<float> B_ = std::vector<float>(bRows * bCols);
  std::vector<float> C_ = std::vector<float>(cRows * cCols);
  std::byte* deviceA_;
  std::byte* deviceB_;
  std::byte* deviceC_;
};

DEFINE_string(device_type, "sysemu", "Device Type to be used (sysemu,fake,silicon)");
DEFINE_uint64(kernel_launch_timeout, 10, "timeout (inseconds) to wait for kernelLaunch");
DEFINE_uint64(num_launches, 1, "Number of times the kernel will be launched");
DEFINE_string(kernel_path, "", "ET-SoC-1 kernel path and filename");
DEFINE_double(epsilon, 0.0, "Delta used for comparison between host and device");

int main(int argc, char** argv) {

  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);
  Config config{modeFromString(FLAGS_device_type), 1};
  config.dump();

  TxFma launcher(config);
  launcher.initialize();
  auto kernelId = launcher.loadKernel(FLAGS_kernel_path);
  launcher.performDeviceAllocs();
  launcher.prepareInput();

  launcher.programHost2DevCopies();

  // prep kernel args object
  Matrix A{launcher.aRows, launcher.aCols, (float*)launcher.deviceA_};
  Matrix B{launcher.bRows, launcher.bCols, (float*)launcher.deviceB_};
  Matrix C{launcher.cRows, launcher.cCols, (float*)launcher.deviceC_};

  KernelArguments kernelArgs{A, B, C};

  launcher.kernelLaunch(kernelId, &kernelArgs);
  launcher.programDev2HostCopies();
  auto timeout = std::chrono::seconds(FLAGS_kernel_launch_timeout);
  launcher.waitKernelCompletion(timeout);
  launcher.dumpTracesToFile();

  if (launcher.kernelError_ || launcher.kernelAbort_) {
    return -1;
  }

  launcher.freeDeviceAllocs();
  launcher.unLoadKernel(kernelId);
  launcher.tearDown();

  // Check results
  std::vector<float> c2 = launcher.C_;
  txfma(c2.data(), launcher.A_.data(), launcher.B_.data(), launcher.aRows, launcher.aCols, launcher.bCols);
  if (!std::equal(c2.begin(), c2.end(), launcher.C_.begin(),
                  [=](float host, float dev) { return std::abs(host - dev) <= FLAGS_epsilon; })) {
    std::cerr << "error: TXFMA host/device results do not match" << std::endl;
    return 1;
  }

  return 0;
}
