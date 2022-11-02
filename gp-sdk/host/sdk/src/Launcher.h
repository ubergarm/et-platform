//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------
#ifndef LAUNCHER_H
#define LAUNCHER_H
#include <device-layer/IDeviceLayer.h>
#include <runtime/IRuntime.h>

// TODO
// FIXME: In case it is needed we should have or own gp-sdk logger,
// although it only affects runtime.
#include <hostUtils/logging/Logger.h>

#include "GenericLauncher.h"
#include <string>
#include <vector>

// Shared with device:
// FIXME: move to a public header.
// bewware of void * and use fixed-with instead.
struct Params {
  void* src1;
  void* src2;
  void* dst;
  int elements;
} __attribute__((packed));



// Specific kernel lancuher class.
class Launcher : public GenericLauncher {
public:
  Launcher() = delete;
  Launcher(const Config& config)
    : GenericLauncher(config){};
  void performDeviceAllocs() override;
  void programHost2DevCopies() override;
  void programDev2HostCopies() override;
  void freeDeviceAllocs() override;
  void prepareKernelArguments() override;

private:
  static constexpr size_t numElems_ = 150;
  std::vector<int> hSrc1_ = std::vector<int>(numElems_);
  std::vector<int> hSrc2_ = std::vector<int>(numElems_);
  std::vector<int> hDst_ = std::vector<int>(numElems_);
  std::byte* dSrc1_;
  std::byte* dSrc2_;
  std::byte* dDst_;
  Params params_;
};

#endif // LAUNCHER_H
