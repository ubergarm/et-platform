
//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <numeric>
#include <string>

#include "GenericLauncher.h"
#include "user_defined_stack_kernel_arguments.h"

/* Place here all parameters accepted for this specific launcher. */
struct Options {

  fs::path kernel_path = "";
  int kernel_launch_timeout = 10;
  int num_launches = 1;
  std::string device_type = "sysemu";
  uint32_t shire_mask = 0xFFFFFFFF;
  uint64_t stackSize = 0;
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
    "  -m, --shire_mask              Shires the kernel will be assigned when executed.\n"
    "  -s, --stackSize               Define the stack size to use for each hart. It has be 4096B aligned.\n";

  static constexpr const char* short_opts = "k:t:n:d:m:s:h";

  static const std::vector<struct option> long_opts_vect{
    {"kernel_path", required_argument, nullptr, 'k'},  {"kernel_launch_timeout", required_argument, nullptr, 't'},
    {"num_launches", required_argument, nullptr, 'n'}, {"device_type", required_argument, nullptr, 'd'},
    {"shire_mask", required_argument, nullptr, 'm'},   {"help", no_argument, nullptr, 'h'},
    {"stackSize", required_argument, nullptr, 's'},    {nullptr, 0, nullptr, 0}};

  Options opts;

  int ret = 0;
  int index = 0;
  opterr = 0;

  while ((ret = getopt_long_only(argc, argv, short_opts, long_opts_vect.data(), &index)) != -1) {
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
    case 's':
      opts.stackSize = atol(optarg);
      break;
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

// Specific kernel launcher class.
class StackLauncher : public GenericLauncher {
public:
  StackLauncher() = delete;
  using GenericLauncher::GenericLauncher;
};

int main(int argc, char** argv) {

  std::vector<char*> argvPendingToParse{argv[0]};

  Options opt = parse_args(argc, argv, argvPendingToParse);

  Config config{modeFromString(opt.device_type), 1};
  config.dump();

  StackLauncher launcher(config, static_cast<int>(argvPendingToParse.size()), argvPendingToParse.data());
  launcher.initialize();
  auto kernelId = launcher.loadKernel(opt.kernel_path);
  auto [ptrStack, totalStackSize] = launcher.allocDeviceStack(opt.stackSize, opt.shire_mask);

  KernelArguments kernelArgs;
  kernelArgs.stackSize = opt.stackSize;

  auto timeout = std::chrono::seconds(opt.kernel_launch_timeout);
  for (size_t i = 0; i < opt.num_launches; i++) {
    launcher.kernelLaunch(kernelId, &kernelArgs, ptrStack, totalStackSize, 0, opt.shire_mask);
    launcher.waitKernelCompletion(timeout);
    launcher.dumpTracesToFile(i);

    if (launcher.checkKernelExecutionErrors()) {
      return -1;
    }
  }

  launcher.freeDeviceStack(ptrStack);
  launcher.unLoadKernel(kernelId);
  launcher.tearDown();

  return 0;
}
