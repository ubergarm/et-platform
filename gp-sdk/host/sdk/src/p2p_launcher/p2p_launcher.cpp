
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
  std::string device_type = "sysemu";
};

Options parse_args(int argc, char* const* argv, std::vector<char*>& nextlevel) {

  std::string launcherName = argv[0];
  static constexpr const char* help_msg =
    "Usage: [options] <trace>\n\n"
    "Launcher GP-SDK kernel.\n\n"
    "The following switches must be given:\n"
    "The following switches are optional:\n"
    "  -d, --device_type             Device Type to be used (sysemu, fake,silicon.\n";

  static constexpr const char* short_opts = "d:";

  static const std::vector<struct option> long_opts_vect{
    {"device_type", required_argument, nullptr, 'd'}, {"help", no_argument, nullptr, 'h'}, {nullptr, 0, nullptr, 0}};

  Options opts;

  int ret = 0;
  int index = 0;
  opterr = 0;

  while ((ret = getopt_long_only(argc, argv, short_opts, long_opts_vect.data(), &index)) != -1) {
    switch (ret) {
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

// Specific kernel launcher class.
struct P2PLauncher : public GenericLauncher {
  P2PLauncher() = delete;
  using GenericLauncher::GenericLauncher;
  void test();
};

void P2PLauncher::test() {

  if (getNumDevices() < 2) {
    std::cout << "Error: at least 2 devices needed for p2p test\n";
    exit(1);
  }

  auto source = devices_[0];
  auto destination = devices_[1];
  if (!runtime_->isP2PEnabled(source, destination)) {
    std::cout << "p2p not enabled betweedn devs " << (uint32_t)source << " and " << (uint32_t)destination << "\n";
    exit(1);
  }

  constexpr size_t size = 1024 * 1024;

  std::vector<uint32_t> cleaner(size / sizeof(uint32_t));
  std::vector<uint32_t> dataIn(size / sizeof(uint32_t));
  std::vector<uint32_t> dataOut(size / sizeof(uint32_t));

  std::fill(cleaner.begin(), cleaner.end(), 0);
  std::iota(dataIn.begin(), dataIn.end(), 0);

  // alloc a buffers on  source  and destination.
  auto sourceDevPtr = runtime_->mallocDevice(source, size);
  auto destinationDevPtr = runtime_->mallocDevice(destination, size);

  // clean the buffers on both devices so we are sure we have no side effects from prev executions.
  auto sourceStream = defaultStreams_[0];
  auto destinationStream = defaultStreams_[1];
  runtime_->memcpyHostToDevice(sourceStream, (std::byte*)cleaner.data(), sourceDevPtr, size);
  runtime_->memcpyHostToDevice(destinationStream, (std::byte*)cleaner.data(), destinationDevPtr, size);
  runtime_->waitForStream(sourceStream);
  runtime_->waitForStream(destinationStream);

  // copy a buffer to source-device
  runtime_->memcpyHostToDevice(sourceStream, (std::byte*)dataIn.data(), sourceDevPtr, size);
  runtime_->waitForStream(sourceStream);

  // p2p copy this buffer from source-device to destination-device
  runtime_->memcpyDeviceToDevice(source, destinationStream, sourceDevPtr, destinationDevPtr, size);

  // copy back from destination-device to the host, (queued into the p2p-copy stream).
  runtime_->memcpyDeviceToHost(destinationStream, destinationDevPtr, (std::byte*)dataOut.data(), size);
  runtime_->waitForStream(destinationStream);

  // check that the loopback copy succeeded.
  if (dataIn != dataOut) {
    std::cout << "Error: buffers do not match, p2p copy failed\n";
    exit(1);
  }

  std::cout << "p2p copy succeeded\n";
}

int main(int argc, char** argv) {

  std::vector<char*> argvPendingToParse{argv[0]};

  Options opt = parse_args(argc, argv, argvPendingToParse);

  Config config{modeFromString(opt.device_type), 1};
  config.dump();

  P2PLauncher launcher(config, static_cast<int>(argvPendingToParse.size()), argvPendingToParse.data());

  launcher.initialize();
  launcher.test();
}
