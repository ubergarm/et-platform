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
#include <numeric>
#include <string>

#include "GenericLauncher.h"

/* Place here all parameters accepted for this specific launcher. */
struct Options {

  fs::path kernel_path = "";
  int kernel_launch_timeout = 10;
  std::string device_type = "sysemu";
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
    "  -d, --device_type             Device Type to be used (sysemu, fake,silicon.\n";

  static constexpr const char* short_opts = "k:t:d:h";

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
class TrappingLauncher : public GenericLauncher {
public:
  TrappingLauncher() = delete;
  using GenericLauncher::GenericLauncher;
};

int main(int argc, char** argv) {
  std::vector<char*> argvPendingToParse{argv[0]};

  Options opt = parse_args(argc, argv, argvPendingToParse);
  Config config{modeFromString(opt.device_type), 1};
  config.dump();

  TrappingLauncher launcher(config, static_cast<int>(argvPendingToParse.size()), argvPendingToParse.data());

  launcher.initialize();
  auto kernelId = launcher.loadKernel(opt.kernel_path);

  launcher.kernelLaunch(kernelId);
  auto timeout = std::chrono::seconds(opt.kernel_launch_timeout);
  launcher.waitKernelCompletion(timeout);
  launcher.dumpTracesToFile(0);

  if(launcher.checkKernelExecutionErrors()) {
    return -1;
  }

  launcher.unLoadKernel(kernelId);
  launcher.tearDown();

  return 0;
}
