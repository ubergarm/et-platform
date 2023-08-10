//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#ifndef RUNTIME_IMP_WITH_CORE_DUMP_H
#define RUNTIME_IMP_WITH_CORE_DUMP_H

#include <AbortManager.h>
#include <runtime/IRuntime.h>

//------------------------------------------------------------------------------------------------
// Runtime wrapper class with core dump capabilities
//------------------------------------------------------------------------------------------------
class RuntimeImpWithCoreDump : public rt::IRuntime {
public:
  RuntimeImpWithCoreDump(rt::IRuntime* runtime, AbortManager* abortManager, bool useRuntimeMultiProcess = false)
    : runtime_(runtime)
    , abortManager_(abortManager)
    , useRuntimeMultiProcess_(useRuntimeMultiProcess) {
  }
  ~RuntimeImpWithCoreDump() final;

  std::vector<rt::DeviceId> doGetDevices() final;
  std::byte* doMallocDevice(rt::DeviceId device, size_t size, uint32_t alignment = rt::kCacheLineSize) final;
  void doFreeDevice(rt::DeviceId device, std::byte* buffer) final;
  rt::StreamId doCreateStream(rt::DeviceId device) final;
  void doDestroyStream(rt::StreamId stream) final;
  rt::LoadCodeResult doLoadCode(rt::StreamId stream, const std::byte* elf, size_t elf_size) final;
  void doUnloadCode(rt::KernelId kernel) final;
  rt::EventId doKernelLaunch(rt::StreamId stream, rt::KernelId kernel, const std::byte* kernel_args,
                             size_t kernel_args_size, uint64_t shire_mask, bool barrier, bool flushL3,
                             std::optional<rt::UserTrace> userTraceConfig, const std::string& coreDumpFilePath) final;
  rt::EventId doMemcpyHostToDevice(rt::StreamId stream, const std::byte* h_src, std::byte* d_dst, size_t size,
                                   bool barrier, const rt::CmaCopyFunction&) final;
  rt::EventId doMemcpyDeviceToHost(rt::StreamId stream, const std::byte* d_src, std::byte* h_dst, size_t size,
                                   bool barrier, const rt::CmaCopyFunction&) final;
  rt::EventId doMemcpyHostToDevice(rt::StreamId stream, rt::MemcpyList memcpyList, bool barrier,
                                   const rt::CmaCopyFunction&) final;
  rt::EventId doMemcpyDeviceToHost(rt::StreamId stream, rt::MemcpyList memcpyList, bool barrier,
                                   const rt::CmaCopyFunction&) final;
  rt::EventId doMemcpyDeviceToDevice(rt::StreamId streamSrc, rt::DeviceId deviceDst, const std::byte* d_src,
                                     std::byte* d_dst, size_t size, bool barrier) final;
  rt::EventId doMemcpyDeviceToDevice(rt::DeviceId deviceSrc, rt::StreamId streamDst, const std::byte* d_src,
                                     std::byte* d_dst, size_t size, bool barrier) final;
  bool doWaitForEvent(rt::EventId event, std::chrono::seconds timeout = std::chrono::hours(24)) final;
  bool doWaitForStream(rt::StreamId stream, std::chrono::seconds timeout = std::chrono::hours(24)) final;
  std::vector<rt::StreamError> doRetrieveStreamErrors(rt::StreamId stream) final;
  void doSetOnStreamErrorsCallback(rt::StreamErrorCallback callback) final;
  rt::DeviceProperties doGetDeviceProperties(rt::DeviceId device) const final;
  void doSetOnKernelAbortedErrorCallback(const rt::KernelAbortedCallback& callback) final;
  rt::EventId doAbortCommand(rt::EventId commandId,
                             std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) final;
  rt::EventId doAbortStream(rt::StreamId streamId) final;
  rt::DmaInfo doGetDmaInfo(rt::DeviceId deviceId) const final;

private:
  rt::IRuntime* runtime_;
  AbortManager* abortManager_;
  bool useRuntimeMultiProcess_;
};

#endif // RUNTIME_WITH_CORE_DUMP_IMP_H
