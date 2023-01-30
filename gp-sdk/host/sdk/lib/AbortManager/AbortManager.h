//******************************************************************************
// Copyright (C) 2018-2023, Esperanto Technologies Inc.
// The copyright to the computer program(s) herein is the
// property of Esperanto Technologies, Inc. All Rights Reserved.
// The program(s) may be used and/or copied only with
// the written permission of Esperanto Technologies and
// in accordance with the terms and conditions stipulated in the
// agreement/contract under which the program(s) have been supplied.
//------------------------------------------------------------------------------

#ifndef ABORT_MANAGER_H
#define ABORT_MANAGER_H

#include <runtime/IRuntime.h>

#include <future>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

//--------------------------------------------------------------------------------------
// Following classes, structures, and constants are extracted from GLOW stack
// and copied with minimal modifications.
//
// TODO: Avoid code duplication over project by creating common "AbortManager" lib
// which would be used by GLOW and GP-SDK.
//--------------------------------------------------------------------------------------
class AbortManager {
public:
  // Type of use for an allocated region of device memory
  enum class DeviceMemoryUse { DATA, CODE, STACK };

  // Properties of a region of allocated device memory
  struct AllocatedDeviceMemory {
    // Size of the area in bytes
    size_t size_;
    // Type of use
    DeviceMemoryUse useType_;
    // If a code region, path of the kernel ELF in the file system
    std::string path_;

    AllocatedDeviceMemory(size_t size, DeviceMemoryUse useType, std::string const& path = "")
      : size_(size)
      , useType_(useType)
      , path_(path) {
    }
  };

  // This class locks the manager and allows to perform queries
  class Handler : public std::unique_lock<std::mutex> {
  private:
    AbortManager& manager_;

  public:
    explicit Handler(AbortManager& manager)
      : std::unique_lock<std::mutex>(manager.mutex_)
      , manager_(manager) {
    }

    // Returns whether there are multiple kernels launched
    bool hasMultipleKernels() const;

    // Returns whether there is code loaded
    bool hasCode() const;

    // Returns the device to which a kernel has been launched
    std::optional<rt::DeviceId> getKernelLaunchDevice(rt::EventId eventId) const;

    // Returns the map of allocated device memory
    std::map<std::byte*, AllocatedDeviceMemory> getAllocatedDeviceMemoryMap(rt::DeviceId deviceId) const {
      std::map<std::byte*, AllocatedDeviceMemory> result;

      for (auto& [location, allocationInfo] : manager_.allocatedDeviceMemory_) {
        if (std::get<1>(location) == deviceId) {
          result.try_emplace(std::get<0>(location), allocationInfo);
        }
      }
      return result;
    }
  };

  // Registers a given region of device memory as being a data region
  void registerData(std::byte* address, size_t size, rt::DeviceId deviceId);
  // Unregisters a device data memory region
  void unregisterData(std::byte* address, rt::DeviceId deviceId);
  // Registers a given region of device memory as a code region
  void registerCode(std::byte* address, size_t size, rt::KernelId kernelId, rt::DeviceId deviceId,
                    std::string const& path = "");
  // Unregisters a device code memory region
  void unregisterCode(rt::KernelId kernelId);

  // Registers that a stream corresponds to a device
  void registerStream(rt::StreamId streamId, rt::DeviceId deviceId);
  // Registers that an kernel is being launched in a stream
  std::promise<rt::EventId> registerKernelLaunch(rt::StreamId streamId, rt::KernelId kernelId);
  // Clear the kernel launches of a stream
  void clearKernelLaunches(rt::StreamId streamId);

  // Checks if any kernel has been launched by the stream
  bool isAnyKernelLaunchedByStream(rt::StreamId streamId);

  // Prepare for aborting a kernel launch. Returns the list of kernel event
  // identifiers that will be aborted
  std::vector<rt::EventId> prepareKernelLaunchAbort(rt::DeviceId deviceId, rt::StreamId streamId,
                                                    rt::IRuntime& runtime);
  // Wait until a kernel launch has been fully aborted
  void waitKernelLaunchAborted(rt::EventId kernelLaunchEventId);
  // Notify that a kernel launch has been fully aborted
  void notifyKernelLaunchAborted(rt::EventId kernelLaunchEventId);
  // Wait until a kernel launch abort callback occurs
  void waitDeviceAbortCallback(rt::EventId kernelLaunchEventId);
  // Notify that a kernel launch abort callback has finished
  void notifyDeviceAbortCallback(rt::EventId kernelLaunchEventId);

  // Get the device identifier used for launching a kernel
  std::optional<rt::DeviceId> getDeviceId(rt::EventId kernelEventId);

  // Get the device identifier from registered stream
  // NOTE: Method doesn't exist in Glow!
  rt::DeviceId getDeviceIdFromStreamId(rt::StreamId streamId);

  // Retrieve the error context for a failed event
  std::optional<rt::StreamError> retrieveErrorContext(rt::IRuntime* runtime, rt::EventId eventId,
                                                      std::byte const* context, size_t size);

  std::optional<rt::StreamError> handleAbortedKernelAndDumpCore(rt::IRuntime* runtime, rt::EventId eventId,
                                                                std::byte const* context, size_t size,
                                                                std::function<void()> freeResources);
  // dump core from a rt::StreamError
  void dumpCore(rt::IRuntime* runtime, rt::EventId eventId, const rt::StreamError& error);

private:
  class KernelLaunchInformation {
  private:
    // Event identifier associated to the kernel launch. Only valid if
    // eventIdFuture_.valid() == false
    rt::EventId eventId_;
    // Future to get the event id. Value will be moved to eventId_
    std::future<rt::EventId> eventIdFuture_;

    // Identifier of the stream on which the kernel is launched
    rt::StreamId streamId_;
    // Identifier of the kernel
    rt::KernelId kernelId_;

  public:
    KernelLaunchInformation(rt::StreamId streamId, rt::KernelId kernelId, std::future<rt::EventId>&& eventIdFuture)
      : eventIdFuture_(std::move(eventIdFuture))
      , streamId_(streamId)
      , kernelId_(kernelId) {
    }

    // Since the event identifier is assigned asyncrhonously, it can only
    // retrieved thorugh a getter
    rt::EventId getEventId() {
      if (eventIdFuture_.valid()) {
        eventId_ = eventIdFuture_.get();
      }
      return eventId_;
    }

    // Get the stream ID
    rt::StreamId getStreamId() const {
      return streamId_;
    }

    // Get the kernel ID
    rt::KernelId getKernelId() const {
      return kernelId_;
    }
  };

  // Aborts require to wait for the queue to be cleared (wait for the abort
  // event), then to proceed with any cleanup (transfer state), then to clean up
  // the runtime (from within the callback), and only after that, possibly to
  // destroy the runtime. This class coordinates these phases.
  struct AbortSync {
    // Condition variable to synchronize between stages
    std::condition_variable condVar_;
    // Runtime object
    rt::IRuntime& runtime_;
    // Device where the abort is being handled
    rt::DeviceId deviceId_;
    // Stream of the kernel launch
    rt::StreamId streamId_;
    // True iff the abort has fully cleared the queue
    bool finishedAbort_ = false;
    // True iff the abort callback has finished
    bool finishedCallback_ = false;

    AbortSync(rt::IRuntime& runtime, rt::DeviceId deviceId, rt::StreamId streamId)
      : runtime_(runtime)
      , deviceId_(deviceId)
      , streamId_(streamId) {
    }
  };

  // Find the kernel launch associated to a given event identifier
  KernelLaunchInformation* findKernelLaunch(rt::EventId eventId);

  // Mutex for the whole object
  std::mutex mutex_;
  // Maps device pointers to the usage and size. Only used if enableCodeDump is
  // true.
  std::map<std::tuple<std::byte*, rt::DeviceId>, AllocatedDeviceMemory> allocatedDeviceMemory_;
  // Maps kernel ids to the element in allocatedDeviceMemory_ that contains
  // their code. Only used if enableCodeDump is true.
  std::map<rt::KernelId, decltype(allocatedDeviceMemory_)::iterator> kernelCodeDeviceMemory_;

  // Maps stream identifiers to the identifier of the devices where they run
  std::map<rt::StreamId, rt::DeviceId> streamToDevice_;

  // Per stream list of kernel launches
  std::map<rt::StreamId, std::list<KernelLaunchInformation>> kernelLaunchesByStream_;

  // Maps kernel launch event identifiers to objects that allow to synchronize
  // the abort
  std::map<rt::EventId, AbortSync> abortSyncronization_;
};

#endif // ABORT_MANAGER_H
