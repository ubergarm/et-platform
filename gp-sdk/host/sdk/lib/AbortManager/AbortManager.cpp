//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#include "AbortManager.h"
#include "CoreDumper.h"

#include <cassert>
#include <fstream>
#include <iostream>

// Returns whether there are multiple kernels launched
bool AbortManager::Handler::hasMultipleKernels() const {
  return (manager_.kernelCodeDeviceMemory_.size() > 1);
}

// Returns whether there is code loaded
bool AbortManager::Handler::hasCode() const {
  return not manager_.kernelCodeDeviceMemory_.empty();
}

// Returns the device to which a kernel has been launched
std::optional<rt::DeviceId> AbortManager::Handler::getKernelLaunchDevice(rt::EventId eventId) const {
  // Find the kernel launch
  KernelLaunchInformation const* kernelLaunch = manager_.findKernelLaunch(eventId);
  if (kernelLaunch == nullptr) {
    // Not a kernel launch error
    return std::optional<rt::DeviceId>();
  }

  // Device for which the core is to be dumped
  return manager_.streamToDevice_[kernelLaunch->getStreamId()];
}

// Registers a given region of device memory as being a data region
void AbortManager::registerData(std::byte* address, size_t size, rt::DeviceId deviceId) {
  std::unique_lock lock(mutex_);

  allocatedDeviceMemory_.emplace(std::piecewise_construct, std::tuple(address, deviceId),
                                 std::forward_as_tuple(size, DeviceMemoryUse::DATA));
}

// Unregisters a device data memory region
void AbortManager::unregisterData(std::byte* address, rt::DeviceId deviceId) {
  std::unique_lock lock(mutex_);

  allocatedDeviceMemory_.erase(std::make_tuple(address, deviceId));
}

// Registers a given region of device memory as a code region
void AbortManager::registerCode(std::byte* address, size_t size, rt::KernelId kernelId, rt::DeviceId deviceId,
                                std::string const& path) {
  std::unique_lock lock(mutex_);

  auto position = allocatedDeviceMemory_.emplace(std::piecewise_construct, std::make_tuple(address, deviceId),
                                                 std::forward_as_tuple(size, DeviceMemoryUse::CODE, path));
  // This is needed to recover the position in allocatedDeviceMemory_ from the
  // kernelId during unregisterCode
  kernelCodeDeviceMemory_[kernelId] = position.first;
}

// Unregisters a device code memory region
void AbortManager::unregisterCode(rt::KernelId kernelId) {
  std::unique_lock lock(mutex_);

  auto kernelPosition = kernelCodeDeviceMemory_.find(kernelId);
  if (kernelPosition != kernelCodeDeviceMemory_.end()) {
    allocatedDeviceMemory_.erase(kernelPosition->second);
    kernelCodeDeviceMemory_.erase(kernelPosition);
  }
}

// Registers that a certain stream corresponds to a device
void AbortManager::registerStream(rt::StreamId streamId, rt::DeviceId deviceId) {
  std::unique_lock lock(mutex_);

  streamToDevice_[streamId] = deviceId;
}

// Registers that a kernel is being launched in a stream
std::promise<rt::EventId> AbortManager::registerKernelLaunch(rt::StreamId streamId, rt::KernelId kernelId) {
  std::promise<rt::EventId> eventIdPromise;

  std::unique_lock lock(mutex_);

  auto& streamKernelLaunches = kernelLaunchesByStream_[streamId];
  streamKernelLaunches.emplace_back(streamId, kernelId, eventIdPromise.get_future());

  return eventIdPromise;
}

// Clear the kernel launches of a stream
void AbortManager::clearKernelLaunches(rt::StreamId streamId) {
  std::unique_lock lock(mutex_);

  auto position = kernelLaunchesByStream_.find(streamId);
  if (position != kernelLaunchesByStream_.end()) {
    kernelLaunchesByStream_.erase(position);
  }
}

// Checks if any kernel has been launched by the stream
bool AbortManager::isAnyKernelLaunchedByStream(rt::StreamId streamId) {
  return (kernelLaunchesByStream_.find(streamId) != kernelLaunchesByStream_.end());
}

// Prepare for aborting a kernel launch. Returns the list of kernel event
// identifiers that will be aborted.
std::vector<rt::EventId> AbortManager::prepareKernelLaunchAbort(rt::DeviceId deviceId, rt::StreamId streamId,
                                                                rt::IRuntime& runtime) {
  std::unique_lock lock(mutex_);

  auto position = kernelLaunchesByStream_.find(streamId);
  assert(position != kernelLaunchesByStream_.end());

  std::vector<rt::EventId> kernelEventIds;
  kernelEventIds.reserve(position->second.size());
  for (auto& kernelLaunch : position->second) {
    auto eventId = kernelLaunch.getEventId();

#ifndef NDEBUG
    auto it = abortSyncronization_.find(eventId);
    assert(it == abortSyncronization_.end());
#endif

    bool finished = runtime.waitForEvent(eventId, std::chrono::seconds(0));
    if (finished) {
      // Skip finished kernels
      continue;
    }

    abortSyncronization_.emplace(std::piecewise_construct, std::make_tuple(eventId),
                                 std::forward_as_tuple(runtime, deviceId, streamId));

    kernelEventIds.push_back(eventId);
  }

  return kernelEventIds;
}

// Wait until a kernel launch has been fully aborted
void AbortManager::waitKernelLaunchAborted(rt::EventId kernelLaunchEventId) {
  std::unique_lock lock(mutex_);

  auto it = abortSyncronization_.find(kernelLaunchEventId);
  assert(it != abortSyncronization_.end());

  auto& sync = it->second;
  sync.condVar_.wait(lock, [&sync] { return sync.finishedAbort_; });
}

// Notify that a kernel launch has been fully aborted
void AbortManager::notifyKernelLaunchAborted(rt::EventId kernelLaunchEventId) {
  std::unique_lock lock(mutex_);

  auto it = abortSyncronization_.find(kernelLaunchEventId);
  assert(it != abortSyncronization_.end());

  auto& sync = it->second;
  assert(not sync.finishedAbort_);
  sync.finishedAbort_ = true;
  sync.condVar_.notify_all();
}

// Wait until a kernel launch abort callback occurs
void AbortManager::waitDeviceAbortCallback(rt::EventId kernelLaunchEventId) {
  std::unique_lock lock(mutex_);

  auto it = abortSyncronization_.find(kernelLaunchEventId);
  assert(it != abortSyncronization_.end());

  auto& sync = it->second;
  sync.condVar_.wait(lock, [&sync] { return sync.finishedCallback_; });
}

// Notify that a kernel launch abort callback has finished
void AbortManager::notifyDeviceAbortCallback(rt::EventId kernelLaunchEventId) {
  std::unique_lock lock(mutex_);

  auto it = abortSyncronization_.find(kernelLaunchEventId);
  assert(it != abortSyncronization_.end());

  auto& sync = it->second;
  assert(not sync.finishedCallback_);
  sync.finishedCallback_ = true;
  sync.condVar_.notify_all();
}

// Get the device identifier used for launching a kernel
std::optional<rt::DeviceId> AbortManager::getDeviceId(rt::EventId kernelEventId) {
  std::unique_lock lock(mutex_);

  // Find the kernel launch
  KernelLaunchInformation const* kernelLaunch = findKernelLaunch(kernelEventId);
  if (kernelLaunch == nullptr) {
    return std::optional<rt::DeviceId>();
  } else {
    return streamToDevice_[kernelLaunch->getStreamId()];
  }
}

// Get the device identifier from registered stream
// NOTE: Method doesn't exist in ETSOCAbortManager
rt::DeviceId AbortManager::getDeviceIdFromStreamId(rt::StreamId streamId) {
  std::unique_lock lock(mutex_);
  return streamToDevice_.at(streamId);
}

// Find the kernel launch associated to a given event identifier
AbortManager::KernelLaunchInformation* AbortManager::findKernelLaunch(rt::EventId eventId) {
  for (auto& [streamId, launches] : kernelLaunchesByStream_) {
    (void)streamId; // Hush compiler warning turned into error
    for (auto& launch : launches) {
      if (launch.getEventId() == eventId) {
        return &launch;
      }
    }
  }

  return nullptr;
}

// Retrieve the error context for a failed event
std::optional<rt::StreamError> AbortManager::retrieveErrorContext(rt::IRuntime* runtime, rt::EventId eventId,
                                                                  std::byte const* context, size_t size) {
  // Used as return value for error cases
  std::optional<rt::StreamError> empty;

  std::unique_lock lock(mutex_);

  // Find the kernel launch
  KernelLaunchInformation const* kernelLaunch = findKernelLaunch(eventId);
  if (kernelLaunch == nullptr) {
    // Not a kernel launch error, so there is no core to generate
    return empty;
  }

  // Device for which the core is to be dumped
  auto deviceId = streamToDevice_[kernelLaunch->getStreamId()];

  rt::StreamError error(rt::DeviceErrorCode::KernelAbortHostAborted, deviceId);

  assert((size % sizeof(rt::ErrorContext)) == 0);
  auto numHarts = size / sizeof(rt::ErrorContext);
  error.errorContext_.emplace(numHarts);

  // Copy the context data into the error context field
  auto copyBackStream = runtime->createStream(deviceId);
  auto copyEventId = runtime->memcpyDeviceToHost(
    copyBackStream, context, reinterpret_cast<std::byte*>(error.errorContext_.value().data()), size, false);
  bool success = runtime->waitForEvent(copyEventId);
  if (not success) {
    std::cout << "Timed out copying core dump data from device.\n";
    return empty;
  }
  runtime->destroyStream(copyBackStream);

  return std::move(error);
}

std::optional<rt::StreamError> AbortManager::handleAbortedKernelAndDumpCore(rt::IRuntime* runtime, rt::EventId eventId,
                                                                            std::byte const* context, size_t size,
                                                                            std::function<void()> freeResources) {
  // Wait until the kernel has been fully aborted so that further commands
  // are not aborted
  waitKernelLaunchAborted(eventId);
  // Retrieve the error context from the device and handle the error
  auto error = retrieveErrorContext(runtime, eventId, context, size);
  if (error.has_value()) {
    CoreDumper::dumpCore(*this, runtime, eventId, error.value());
  } else {
    std::cout << "[ERROR] Device error (core dump not enabled)\n";
  }

  // Release the runtime resources before allowing the aborter thread to
  // continue
  freeResources();

  // Allow the aborter thread to continue
  notifyDeviceAbortCallback(eventId);

  std::cout << "[FATAL ERROR] Kernel aborted. GP SDK cannot recover from this, "
               "finishing the execution\n";

  return error;
}

void AbortManager::dumpCore(rt::IRuntime* runtime, rt::EventId eventId, const rt::StreamError& error) {
  // FIXME. check if needed
  // waitKernelLaunchAborted(eventId);
  if (error.errorContext_.has_value()) {
    CoreDumper::dumpCore(*this, runtime, eventId, error);
  } else {
    std::cout << "[ERROR] Device error (core dump not enabled)\n";
  }
  // Allow the aborter thread to continue
  // notifyDeviceAbortCallback(eventId);

  std::cout << "[FATAL ERROR] Kernel aborted. GP SDK cannot recover from this, "
               "finishing the execution\n";
}
