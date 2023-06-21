//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>

#include "RuntimeImpWithCoreDump.h"

using namespace std::literals::chrono_literals;

std::vector<rt::DeviceId> RuntimeImpWithCoreDump::doGetDevices() {
  return this->runtime_->getDevices();
}

std::byte* RuntimeImpWithCoreDump::doMallocDevice(rt::DeviceId device, size_t size, uint32_t alignment) {
  auto ptr = this->runtime_->mallocDevice(device, size, alignment);
  if (ptr) {
    abortManager_->registerData(ptr, size, device);
  }
  return ptr;
}

void RuntimeImpWithCoreDump::doFreeDevice(rt::DeviceId device, std::byte* buffer) {
  this->runtime_->freeDevice(device, buffer);
  abortManager_->unregisterData(buffer, device);
}

rt::StreamId RuntimeImpWithCoreDump::doCreateStream(rt::DeviceId device) {
  rt::StreamId streamId = this->runtime_->createStream(device);
  abortManager_->registerStream(streamId, device);
  return streamId;
}

void RuntimeImpWithCoreDump::doDestroyStream(rt::StreamId stream) {
  this->runtime_->destroyStream(stream);
}

rt::LoadCodeResult RuntimeImpWithCoreDump::doLoadCode(rt::StreamId stream, const std::byte* elf, size_t elf_size) {
  auto res = this->runtime_->loadCode(stream, elf, elf_size);
  runtime_->waitForEvent(res.event_);

  if (res.loadAddress_) {
    auto deviceId = abortManager_->getDeviceIdFromStreamId(stream);
    abortManager_->registerCode(res.loadAddress_, elf_size, res.kernel_, deviceId);
  }
  return res;
}

void RuntimeImpWithCoreDump::doUnloadCode(rt::KernelId kernel) {
  this->runtime_->unloadCode(kernel);
  abortManager_->unregisterCode(kernel);
}

rt::EventId RuntimeImpWithCoreDump::doKernelLaunch(rt::StreamId stream, rt::KernelId kernel,
                                                   const std::byte* kernel_args, size_t kernel_args_size,
                                                   uint64_t shire_mask, bool barrier, bool flushL3,
                                                   std::optional<rt::UserTrace> userTraceConfig) {

  // This promise is used to avoid a race condition where the RT responds with
  // an abort before the core dumper has registered the event id
  std::promise<rt::EventId> promisedEventId;
  promisedEventId = abortManager_->registerKernelLaunch(stream, kernel);

  auto eventId = this->runtime_->kernelLaunch(stream, kernel, kernel_args, kernel_args_size, shire_mask, barrier, flushL3,
                                      userTraceConfig);

  promisedEventId.set_value(eventId);

  return eventId;
}

rt::EventId RuntimeImpWithCoreDump::doMemcpyHostToDevice(rt::StreamId stream, const std::byte* h_src, std::byte* d_dst,
                                                         size_t size, bool barrier,
                                                         const rt::CmaCopyFunction& cmaCopyFunction) {
  return this->runtime_->memcpyHostToDevice(stream, h_src, d_dst, size, barrier, cmaCopyFunction);
}

rt::EventId RuntimeImpWithCoreDump::doMemcpyDeviceToHost(rt::StreamId stream, const std::byte* d_src, std::byte* h_dst,
                                                         size_t size, bool barrier,
                                                         const rt::CmaCopyFunction& cmaCopyFunction) {
  return this->runtime_->memcpyDeviceToHost(stream, d_src, h_dst, size, barrier, cmaCopyFunction);
}

rt::EventId RuntimeImpWithCoreDump::doMemcpyHostToDevice(rt::StreamId stream, rt::MemcpyList memcpyList, bool barrier,
                                                         const rt::CmaCopyFunction& cmaCopyFunction) {
  return this->runtime_->memcpyHostToDevice(stream, memcpyList, barrier, cmaCopyFunction);
}

rt::EventId RuntimeImpWithCoreDump::doMemcpyDeviceToHost(rt::StreamId stream, rt::MemcpyList memcpyList, bool barrier,
                                                         const rt::CmaCopyFunction& cmaCopyFunction) {
  return this->runtime_->memcpyDeviceToHost(stream, memcpyList, barrier, cmaCopyFunction);
}

rt::EventId RuntimeImpWithCoreDump::doMemcpyDeviceToDevice(rt::StreamId streamSrc, rt::DeviceId deviceDst,
                                                           const std::byte* d_src, std::byte* d_dst, size_t size,
                                                           bool barrier) {
  return this->runtime_->memcpyDeviceToDevice(streamSrc, deviceDst, d_src, d_dst, size, barrier);
}

rt::EventId RuntimeImpWithCoreDump::doMemcpyDeviceToDevice(rt::DeviceId deviceSrc, rt::StreamId streamDst,
                                                           const std::byte* d_src, std::byte* d_dst, size_t size,
                                                           bool barrier) {
  return this->runtime_->memcpyDeviceToDevice(deviceSrc, streamDst, d_src, d_dst, size, barrier);
}

bool RuntimeImpWithCoreDump::doWaitForEvent(rt::EventId event, std::chrono::seconds timeout) {
  return this->runtime_->waitForEvent(event, timeout);
}

bool RuntimeImpWithCoreDump::doWaitForStream(rt::StreamId stream, std::chrono::seconds timeout) {
  auto success = this->runtime_->waitForStream(stream, timeout);
  if(success){
    abortManager_->clearKernelLaunches(stream);
  }
  return success;
}

std::vector<rt::StreamError> RuntimeImpWithCoreDump::doRetrieveStreamErrors(rt::StreamId stream) {
  return this->runtime_->retrieveStreamErrors(stream);
}

void RuntimeImpWithCoreDump::doSetOnStreamErrorsCallback(rt::StreamErrorCallback callback) {
  this->runtime_->setOnStreamErrorsCallback(callback);
}

rt::DeviceProperties RuntimeImpWithCoreDump::doGetDeviceProperties(rt::DeviceId device) const {
  return this->runtime_->getDeviceProperties(device);
}

void RuntimeImpWithCoreDump::doSetOnKernelAbortedErrorCallback(const rt::KernelAbortedCallback& callback) {
  this->runtime_->setOnKernelAbortedErrorCallback(callback);
}

rt::EventId RuntimeImpWithCoreDump::doAbortCommand(rt::EventId commandId, std::chrono::milliseconds timeout) {
  return this->runtime_->abortCommand(commandId, timeout);
}

rt::EventId RuntimeImpWithCoreDump::doAbortStream(rt::StreamId streamId) {
  std::vector<rt::EventId> abortedKernelLaunchEventIds;
  abortedKernelLaunchEventIds =
    abortManager_->prepareKernelLaunchAbort(abortManager_->getDeviceIdFromStreamId(streamId), streamId, *runtime_);

  
  auto event = this->runtime_->abortStream(streamId);
  
  // Wait for the abort to complete.
  auto abortTimeout = std::chrono::seconds(10);
  auto success = runtime_->waitForEvent(event, abortTimeout);
  
  if (success) {
    // Allow the callbacks to proceed
    for (auto const& kernelEventId : abortedKernelLaunchEventIds) {
      abortManager_->notifyKernelLaunchAborted(kernelEventId);
    }

    // Wait until the callbacks have been received to avoid destructing the
    // runtime too early
    for (auto const& kernelEventId : abortedKernelLaunchEventIds) {
      abortManager_->waitDeviceAbortCallback(kernelEventId);
    }

    std::cout << "[        ] " << __func__ << "() stream aborted correctly \n" << int(abortManager_->getDeviceIdFromStreamId(streamId)) << "\n";

  } else {
    // could not complete the abort.
    std::cout << "[TIMEOUT] " << __func__ << "() timeout aborting stream \n" << int(abortManager_->getDeviceIdFromStreamId(streamId)) << "\n";
  }

  return event;
}

rt::DmaInfo RuntimeImpWithCoreDump::doGetDmaInfo(rt::DeviceId deviceId) const {
  return this->runtime_->getDmaInfo(deviceId);
}

RuntimeImpWithCoreDump::~RuntimeImpWithCoreDump() {
}

