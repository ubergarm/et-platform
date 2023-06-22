/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef GPSDK_FLB_H
#define GPSDK_FLB_H

#include "entryPoint.h"
#include <cstddef>
#include <etsoc/isa/atomic.h>


#define REP4(...) {__VA_ARGS__}, {__VA_ARGS__}, {__VA_ARGS__}, {__VA_ARGS__}
#define REP16(...) REP4(__VA_ARGS__),  REP4(__VA_ARGS__), REP4(__VA_ARGS__), REP4(__VA_ARGS__)
#define REP32(...) REP16(__VA_ARGS__), REP16(__VA_ARGS__)


#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

/// @brief This class manages the access to the 32 FLB registers available per shire. A lock should be instantiated per shire.
class flbLock {
  private: 
    enum class lock_status : uint32_t { locked = 0, available = 1 };

    typedef struct alignas(CACHE_LINE_SIZE) {
      lock_status status; // 
      uint32_t begin; // first thread 
      uint32_t end; // last thread 
    } flblock_t;

    static inline flblock_t __flbs[32][32] = { REP32(REP32(lock_status::available, 0, 0)) };
                           
    /// @brief Joins an existing lock. It checks if \p threadId is part of the barrier using flb#<\p flbId>
    /// @param lck FLB register used for sync, values 0 to 31.
    /// @param threadId thread id corresponding to the minion trying to join the barrier, passed as a parameter to avoid extra syscalls.
    /// @return Returns true if the FLB is actively used (locked) in a barrier() and \p threadId is between the begin and end range of threads.
    inline bool join(const flblock_t * lck, const int threadId) {
      const bool isLocked = (getStatus(lck) == lock_status::locked);
      const bool inRange = (static_cast<uint32_t>(threadId) >= getBegin(lck)) && (static_cast<uint32_t>(threadId) <= getEnd(lck));
      return isLocked && inRange;
    }

    // Get and set: all accessed to L2
    inline auto getBegin(const flblock_t * lck) const -> uint32_t {
      return atomic_load_local_32(&(lck->begin));
    }

    inline auto getEnd(const flblock_t * lck) const -> uint32_t {
      return atomic_load_local_32(&(lck->end));
    }

    inline auto getStatus(const flblock_t * lck) const -> lock_status {
      return static_cast<lock_status>(atomic_load_local_32((const uint32_t *) &(lck->status)));
    }

    inline void setRange(flblock_t * lck, const int begin, const int end) {
      atomic_store_local_32(&(lck->begin), static_cast<uint32_t>(begin));
      atomic_store_local_32(&(lck->end), static_cast<uint32_t>(end));
    }

  public:
    flbLock() = default;

    /// @brief Acquires a lock. If the lock is not being used, it assigns the range [begin,end] of threads that can access it.
    /// @param shireId id of shire accessing the lock
    /// @param flbId id of the flb to lock
    /// @param threadId thread id that wants to acquire the lock.
    /// @param begin first thread of the range of threads that can acquire the lock.
    /// @param end  last thread that can acquire the lock.
    /// @return 
    inline bool acquire(size_t shireId, size_t flbId, uint32_t threadId, int begin, int end) {
      bool updateRange = true;

      // address of the lock
      auto lckAddr = &__flbs[shireId][flbId];

      // loop until lock becomes available; then acquire (set to 'locked' again) the lock.
      while (atomic_exchange_local_32((uint32_t *) &(lckAddr->status), (uint32_t) lock_status::locked) != (uint32_t) lock_status::available) {
        asm volatile("fence\n" ::: "memory");

        // If a thread is part of flb#<flbId> range of threads, exit the loop

        if (join(lckAddr, threadId)) {
          updateRange = false;
          break;
        }
      }

      // only the first thread set ups the range
      if (updateRange) {
        setRange(lckAddr, begin, end);
      }

      return true;
    } 

    /// @brief Releases a lock. It should be invoked only once. Only threads in the range assigned to the lock can do the release.
    /// @param shireId id of the shire acessing th elock
    /// @param flbId if do the flb to lock
    /// @param threadId thread id that wants to release the lock.
    /// @return returns true if the lock is released.
    bool release(size_t shireId, size_t flbId, uint32_t threadId) {
      auto lckAddr = &__flbs[shireId][flbId];

      if (join(lckAddr, threadId)) {
        atomic_store_local_32((uint32_t *) &(lckAddr->status), (uint32_t) lock_status::available);
        return true;
      }
      else {
        et_assert(false && "Thread does not have permission to release the lock.");
        return false;
      }
    }
 };

#endif
  


