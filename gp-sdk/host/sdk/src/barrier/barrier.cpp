//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#include <cstdlib>
#include <getopt.h>
#include <string>
#include <numeric>

#include "GenericLauncher.h"
#include "barrierKernelArguments.h"

static constexpr size_t numElements = 2048;

/* Place here all parameters accepted for this specific launcher. */
struct Options {

  fs::path kernel_path = "";
  int kernel_launch_timeout = 10;
  int num_launches = 1;
  std::string device_type = "sysemu";
  uint32_t shire_mask = 0xFFFFFFFF;
};

Options parse_args(int argc, char* const* argv, std::vector<char*>& nextlevel) {

  std::string launcherName = argv[0];
  static constexpr const char* help_msg =
    "Usage: [options] <trace>\n\n"
    "Launcher GP-SDK kernel.\n\n"
    "The following switches must be given:\n"
    "  -k, --kernel_path             path to kernel elf file to execute.\n\n"
    "The following switches are optional:\n"
    "  -t, --kernel_launch_timeout   timeout (in seconds) to wait for kenelLaunch\n"
    "  -n, --num_launches            Number of times the kernel will be launched.\n"
    "  -d, --device_type             Device Type to be used (sysemu, fake,silicon.\n"
    "  -m, --shire_mask              Shires the kernel will be assigned when executed.\n";

  static constexpr const char* short_opts = "k:t:n:d:m:h";

  static const std::vector<struct option> long_opts_vect{{"kernel_path", required_argument, nullptr, 'k'},
                                                         {"kernel_launch_timeout", required_argument, nullptr, 't'},
                                                         {"num_launches", required_argument, nullptr, 'n'},
                                                         {"device_type", required_argument, nullptr, 'd'},
                                                         {"shire_mask", required_argument, nullptr, 'm'},
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
    case 'n':
      opts.num_launches = atoi(optarg);
      break;
    case 'd':
      opts.device_type = optarg;
      break;
    case 'm':
      opts.shire_mask = std::stoul(optarg, 0, 16);
      break;
    case 'h':
      std::cout << help_msg << GenericLauncher::help_msg << std::endl;
      exit(0);
    case '?':
      nextlevel.emplace_back(argv[optind - 1]);
      break;
    default:
      std::cout << "Error: Unknown option " << argv[optind - 1] << ". See " << argv[0] << " --help'.\n" << std::endl;
      exit(1);
    }
  }

  return opts;
}

// Specific kernel lancuher class.
class BarrierLauncher : public GenericLauncher {
public:
  BarrierLauncher() = delete;
  using GenericLauncher::GenericLauncher;

  void performDeviceAllocs() {
    deviceData_ = runtime_->mallocDevice(devices_[devIdx_], data_.size() * sizeof(uint64_t));
    deviceAccumData_ = runtime_->mallocDevice(devices_[devIdx_], accumData_.size() * sizeof(uint64_t));
  }

  void programHost2DevCopies() {
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte *) data_.data(), deviceData_,
                                 data_.size() * sizeof(uint64_t));
    runtime_->memcpyHostToDevice(defaultStreams_[devIdx_], (std::byte *)  accumData_.data(), deviceAccumData_,
                                 accumData_.size() * sizeof(uint64_t));
  }

  void freeDeviceAllocs() {
    runtime_->freeDevice(devices_[devIdx_], deviceData_);
    runtime_->freeDevice(devices_[devIdx_], deviceAccumData_);
  }

  std::vector<uint64_t> data_ = std::vector<uint64_t>(numElements, 0);
  std::vector<uint64_t> accumData_ = std::vector<uint64_t>(numElements, 0);
  std::byte* deviceData_;
  std::byte* deviceAccumData_;
};

int main(int argc, char** argv) {

  std::vector<char*> argvPendingToParse{argv[0]};

  Options opt = parse_args(argc, argv, argvPendingToParse);
  Config config{modeFromString(opt.device_type), 1};
  config.dump();

  BarrierLauncher launcher(config, static_cast<int>(argvPendingToParse.size()), argvPendingToParse.data());
  launcher.initialize();
  auto kernelId = launcher.loadKernel(opt.kernel_path);
  launcher.performDeviceAllocs();

  KernelArguments kernelArgs;
  kernelArgs.data = (uint64_t *) launcher.deviceData_;
  kernelArgs.accumData = (uint64_t *) launcher.deviceAccumData_;

  for (size_t i = 0; i < opt.num_launches; i++) {
    launcher.programHost2DevCopies();
    launcher.kernelLaunch(kernelId, &kernelArgs, 0, opt.shire_mask);

    // launcher.programDev2HostCopies();
    auto timeout = std::chrono::seconds(opt.kernel_launch_timeout);
    launcher.waitKernelCompletion(timeout);
    launcher.dumpTracesToFile(i);

    if(launcher.checkKernelExecutionErrors()) {
      return -1;
    }
  }

  launcher.freeDeviceAllocs();
  launcher.unLoadKernel(kernelId); 
  launcher.tearDown();

  return 0;
}
