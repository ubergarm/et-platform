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
    static constexpr size_t numFlbs = 32;
    static constexpr size_t reservedFlbs = 2; // FLB 0 and 1 are reserved for barriers fast-path
    static constexpr size_t numShires = 32;
    
    typedef struct alignas(CACHE_LINE_SIZE) {
      lock_status status; // 
      uint32_t begin; // first thread 
      uint32_t end; // last thread 
      uint32_t threadsToSync;
    } flblock_t;

    static inline flblock_t __flbs[numShires][numFlbs] = { REP32(REP32(lock_status::available, 0, 0, 0)) };
    static inline flblock_t __mainLock[numShires] = { REP32(lock_status::available, 0, 0, 0) };
        
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

    inline auto getThreadsToSync(const flblock_t * lck) const -> uint32_t {
      return atomic_load_local_32(&(lck->threadsToSync));
    }

    inline auto setRange(flblock_t * lck, const uint32_t begin, const uint32_t end, const uint32_t threadsToSync) -> void {
      atomic_store_local_32(&(lck->begin), begin);
      atomic_store_local_32(&(lck->end), end);
      atomic_store_local_32(&(lck->threadsToSync), threadsToSync);
    }

    inline auto spinLock(lock_status * lck) -> void {
      while (atomic_exchange_local_32((uint32_t *) lck, (uint32_t) lock_status::locked) != (uint32_t) lock_status::available) {
        asm volatile("fence\n" ::: "memory");
      }
    }
    
    inline auto setLock(lock_status * lck) -> uint32_t {
      uint32_t res = atomic_exchange_local_32((uint32_t *) lck, (uint32_t) lock_status::locked);
      asm volatile("fence\n" ::: "memory");
      return res;
    }

    inline auto setAvail(lock_status * lck) -> uint32_t {
      uint32_t res = atomic_exchange_local_32((uint32_t *) lck, (uint32_t) lock_status::available);
      asm volatile("fence\n" ::: "memory");
      return res;
    }

    /// @brief Tries to join an flb lock. A lock is identified (primary key) by the tuple {begin, end, threadsToSync}. If threadId 
    /// @param shireId ID of the shire where threads sync
    /// @param threadId ID of the thread that wants to join the barrier
    /// @param begin First thread of the barrier
    /// @param end Last thread of the barrier
    /// @param threadsToSync number of threads per minion to sync in the barrier
    /// @return returns a valid flb id if the thread belongs to a previous existing lock, returns -1 otherwise
    inline auto joinAny(const size_t shireId, const uint32_t threadId, const uint32_t begin, const uint32_t end, uint32_t threadsToSync) -> int64_t {
      for (size_t i = reservedFlbs; i < numFlbs; i++) {
        auto lckAddr = &__flbs[shireId][i];

        const bool sameBegin = getBegin(lckAddr) == begin;
        const bool sameEnd = getEnd(lckAddr) == end;
        const bool sameTTS = getThreadsToSync(lckAddr) == threadsToSync;
        const bool isLocked = (getStatus(lckAddr) == lock_status::locked);
        const bool inRange = (static_cast<uint32_t>(threadId) >= getBegin(lckAddr)) && (static_cast<uint32_t>(threadId) <= getEnd(lckAddr));

        if(sameBegin && sameEnd && isLocked && inRange && sameTTS)
          return i;
      }
      return -1;
    }

  public:
    flbLock() = default;

    /// @brief Acquires the first available lock, if there are no locks available performs an active wait.
    /// @param shireId id of shire accessing the lock
    /// @param threadId thread id that wants to acquire the lock.
    /// @param begin first thread of the range of threads that can acquire the lock.
    /// @param end  last thread that can acquire the lock.
    /// @return returns a valid flb id if the thread belongs to a previous existing lock, returns -1 otherwise
    inline auto acquireAny(uint32_t shireId, uint32_t threadId, uint32_t begin, uint32_t end, uint32_t threadsToSync) -> int64_t {
      spinLock(&(__mainLock[shireId].status));

      auto flbId = joinAny(shireId, threadId, begin, end, threadsToSync);
      while (flbId < 0) {
        // loop until an flb is available
        for (size_t i = reservedFlbs; i < numFlbs; i++) {
          auto lckAddress = &__flbs[shireId][i];
          if (getStatus(lckAddress) == lock_status::available) {
            // et_printf("Found avail [S:%u][T:%u] -> [FLB:%lu, b:%u, e:%u]\n", shireId, threadId, i, begin, end);
            setLock(&(lckAddress->status));
            setRange(lckAddress, begin, end, threadsToSync);
            flbId = i;
            break;
          }
        }
      }
      // et_printf("I joined [S:%u][T:%u]->[FLB:%lu]\n", shireId, threadId, flbId);
      setAvail(&(__mainLock[shireId].status));
      return flbId;
    }
      
    /// @brief Releases a lock. It should be invoked only once. Only threads in the range assigned to the lock can do the release.
    /// @param shireId id of the shire acessing th elock
    /// @param flbId if do the flb to lock
    /// @param threadId thread id that wants to release the lock.
    /// @return returns true if the lock is released.
    bool release(size_t shireId, size_t flbId, uint32_t threadId, uint32_t begin, uint32_t end, uint32_t threadsToSync) {
      auto lckAddr = &__flbs[shireId][flbId];

      if (joinAny(shireId, threadId, begin, end, threadsToSync) == static_cast<int64_t>(flbId)) {
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
  


