//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#ifdef FFT_VERIFICATION
#include <fftw3.h>
#endif

#include <cstdlib>
#include <getopt.h>
#include <string>

#include "GenericLauncher.h"
// Shared kernel-arguments with device:
#include "kernel_arguments.h"

static constexpr int32_t numThreads = 32;

/* Place here all parameters accepted for this specific launcher. */
struct Options {

  fs::path kernel_path = "";
  int kernel_launch_timeout = 10;
  std::string device_type = "sysemu";
};

Options parse_args(int argc, char* const* argv) {

  std::string launcherName = argv[0];
  static constexpr const char* help_msg =
    "Usage: [options] <trace>\n\n"
    "Launcher GP-SDK kernel.\n\n"
    "The following switches must be given:\n"
    "  -k, --kernel_path             path to kernel elf file to execute.\n\n"
    "The following switches are optional:\n"
    "  -t, --kernel_launch_timeout   timeout (in seconds) to wait for kenelLaunch\n"
    "  -d, --device_type             Device Type to be used (sysemu, fake,silicon.\n";

  static constexpr const char* short_opts = "k:t:n:d:h";

  static const std::vector<struct option> long_opts_vect{{"kernel_path", required_argument, nullptr, 'k'},
                                                         {"kernel_launch_timeout", required_argument, nullptr, 't'},
                                                         {"device_type", required_argument, nullptr, 'd'},
                                                         {"help", no_argument, nullptr, 'h'},
                                                         {nullptr, 0, nullptr, 0}};

  Options opts;

  int ret = 0;
  int index = 0;
  opterr = 0;

  while ((ret = getopt_long(argc, argv, short_opts, long_opts_vect.data(), &index)) != -1) {
    switch (ret) {
    case 'k':
      opts.kernel_path = optarg;
      break;
    case 't':
      opts.kernel_launch_timeout = atoi(optarg);
      break;
    case 'd':
      opts.device_type = optarg;
      break;
    case 'h':
      std::cout << help_msg << GenericLauncher::help_msg << std::endl;
      exit(0);
    case '?':
      break;
    default:
      std::cout << "Error: Unknown option " << argv[optind - 1] << ". See " << argv[0] << " --help'.\n" << std::endl;
      exit(1);
    }
  }

  return opts;
}

// dnnlib FFT kernel lancuher class.
class FFTLauncher : public GenericLauncher {
public:
  FFTLauncher() = delete;
  using GenericLauncher::GenericLauncher;
  /// \brief fill input vector with data. if we are doing an FFT, only real data is in use. Im planes are 0-ed.
  void prepareInput() {

    auto planeSize = height_ * width_;

    std::fill(inputTensor_.begin(), inputTensor_.end(), 0);
    // producing random inputs. Implementations may choose to enter real data here (e.g. open jpeg or png files).
    for (size_t b = 0; b < batch_; b++) {
      for (size_t c = 0; c < channels_; c++) {
        auto base = planeSize * planes_ * b * c;
        auto start = inputTensor_.begin() + base;
        auto end = start + planeSize;
        std::generate(start, end, [] { return rand() % 255; });
      }
    }
  }

#ifdef FFT_VERIFICATION
  /// \brief gets a std::vector with the golden fft results using fftw3 lib on host.
  /// @param batchId batch of images where to get the FFT plane
  /// @param channelId Channel (R,G,B) where the plane is obtained.
  /// @return flatened vector containing the fft-plane. (1st-half real, 2nd-half img)
  std::vector<float> getGoldenFFT(size_t batchId, size_t channelId) {
    assert(operation_ == FFTOp::FFT);

    auto planeSize = height_ * width_;
    auto in = fftwf_alloc_complex(planeSize);
    auto out = fftwf_alloc_complex(planeSize);
    auto plan = fftwf_plan_dft_2d(height_, width_, in, out, operation_ == FFTOp::FFT ? FFTW_FORWARD : FFTW_BACKWARD,
                                  FFTW_ESTIMATE);

    // fill in with data. (just 1 channel from the input tensor)
    auto base = planeSize * planes_ * batchId * channelId;
    for (size_t i = 0; i < height_ * width_; i++) {
      in[i][0] = inputTensor_[base + i];
      in[i][1] = 0;
    }

    // execute the fft on the host
    fftwf_execute(plan);
    // normalize and reformat.
    std::vector<float> outputVector(planeSize * planes_);

    for (size_t i = 0; i < planeSize; i++) {
      auto normalizationFactor = float(1) / float(planeSize);
      outputVector[i] = out[i][0] * normalizationFactor;
      outputVector[i + planeSize] = out[i][1] * normalizationFactor;
    }

    fftwf_destroy_plan(plan);
    fftwf_cleanup();
    fftwf_free(in);
    fftwf_free(out);

    return outputVector;
  }

  /// \brief compares element-wise 2 vectors (absolute and relative error). using a passed epsilon.
  ///@param v1: plane to compare
  ///@param v2: full output vector
  ///@param batchId_: batch_ to compare from outputTensor_
  ///@param chanelId channel to compare from outputTenor:
  ///@return wether vectors are equivalent (within epsilon range).
  bool comparePlane(const std::vector<float>& v1, const std::vector<float>& v2, size_t batchId, size_t channelId,
                    float epsilon = 0.01) {
    auto base = width_ * height_ * planes_ * batchId * channelId;
    return std::equal(v1.begin(), v1.end(), v2.begin() + base, [epsilon](const float& f1, const float& f2) {
      if (std::abs(f1) - std::abs(f2) < epsilon) {
        return true;
      }
      return std::abs(1 - (std::abs(f1) / std::abs(f2 + 0.000001))) < epsilon;
    });
  }

  void verify() {
    // check results plane by plane:
    for (size_t b = 0; b < batch_; b++) {
      for (size_t c = 0; c < channels_; c++) {
        auto fft = getGoldenFFT(b, c);
        auto equal = comparePlane(fft, outputTensor_, b, c);
        std::string passStr = equal ? "OK" : "FAIL";
        std::cout << "fft verification: " << b << " " << c << " " << passStr << "\n";
      }
    }
  }

#endif

  void performDeviceAllocs() {
    devInputTensor = runtime_->mallocDevice(devices_[devIdx_], inputTensor_.size() * sizeof(float));
    devOutputTensor = runtime_->mallocDevice(devices_[devIdx_], outputTensor_.size() * sizeof(float));
  }

  void programHost2DevCopies() {
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte*)inputTensor_.data(), devInputTensor,
                                 inputTensor_.size() * sizeof(float));
  }

  void programDev2HostCopies() {
    runtime_->memcpyDeviceToHost(defaultStreams_[devIdx_], devOutputTensor, (std::byte*)outputTensor_.data(),
                                 outputTensor_.size() * sizeof(float));
  }

  void freeDeviceAllocs() {
    runtime_->freeDevice(devices_[devIdx_], devInputTensor);
    runtime_->freeDevice(devices_[devIdx_], devOutputTensor);
  }

  // FFT computation dimensions, it assumes a flattened tensor of 5 dims:
  // batch, channels, planes (2), height_, width_;
  // as an example, we can store a batch of 2 rgb images (feq domain).
  // batch=2 (im0, im1), channels = 3 (r,g,b) , planes = 2(real, img), height_ width_ (256 * 256 pixels images)
  static constexpr size_t height_ = 16;  //!< nedds to be power of 2.
  static constexpr size_t width_ = 16;   //!< needs to be power of 2
  static constexpr size_t planes_ = 2;   //!< only 2 is valid (real and imaginary planes).
  static constexpr size_t channels_ = 1; //!< number of image channels.
  static constexpr size_t batch_ = 1;    //!< number of images.
  static constexpr size_t elements_ = batch_ * channels_ * planes_ * width_ * height_;
  std::vector<float> inputTensor_ = std::vector<float>(elements_);
  std::vector<float> outputTensor_ = std::vector<float>(elements_);
  FFTOp operation_ = FFTOp::FFT;
  std::byte* devInputTensor = nullptr;
  std::byte* devOutputTensor = nullptr;
  KernelArguments fftKernelArgs_;
};

int main(int argc, char** argv) {
  Options opt = parse_args(argc, argv);
  Config config{modeFromString(opt.device_type), 1};
  config.dump();

  FFTLauncher launcher(config, argc, argv);
  launcher.initialize();
  auto kernelId = launcher.loadKernel(opt.kernel_path);
  launcher.prepareInput();
  launcher.performDeviceAllocs();

  launcher.programHost2DevCopies();
  
  // Prepare kernel arguments & launch the kernel.
  TensorDesc inputTensorDesc = {5, {launcher.batch_, launcher.channels_, launcher.planes_, launcher.height_, launcher.width_},
				{launcher.width_ * launcher.height_ * launcher.planes_ * launcher.channels_,
				 launcher.width_ * launcher.height_ * launcher.planes_, launcher.width_ * launcher.height_, launcher.width_, 1},
				uint64_t(launcher.devInputTensor)};
  TensorDesc outputTensorDesc = {5, {launcher.batch_, launcher.channels_, launcher.planes_, launcher.height_, launcher.width_},
				 {launcher.width_ * launcher.height_ * launcher.planes_ * launcher.channels_,
				  launcher.width_ * launcher.height_ * launcher.planes_, launcher.width_ * launcher.height_, launcher.width_, 1},
				 uint64_t(launcher.devOutputTensor)};
 

  KernelArguments kernelArgs;
  kernelArgs.nTensors = 2;
  kernelArgs.tensors[0] = inputTensorDesc;
  kernelArgs.tensors[1] = outputTensorDesc;
  kernelArgs.operation = uint32_t(launcher.operation_);

  launcher.kernelLaunch(kernelId, &kernelArgs);
  launcher.programDev2HostCopies();

  auto timeout = std::chrono::seconds(opt.kernel_launch_timeout);
  launcher.waitKernelCompletion(timeout);
  launcher.dumpTracesToFile();
  
  launcher.freeDeviceAllocs();
  launcher.unLoadKernel(kernelId); 
  launcher.tearDown();

#ifdef FFT_VERIFICATION
  launcher.verify();
#endif

  return 0;
}
