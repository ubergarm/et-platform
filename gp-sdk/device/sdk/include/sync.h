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

#ifndef GPSDK_SYNC_H
#define GPSDK_SYNC_H

#include <bitset>
#include <type_traits>

// Device common
#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

// FW syscall IDs
#include <etsoc/isa/syscall.h>
#include "system/abi.h"

#include <profiling.h>

#include "entryPoint.h"

namespace device_config {
extern DeviceConfig config;
extern const kernel_environment_t * env_;
}

/// @brief Obtains the relative minion id where the relative thread id is located
/// @param relative_thread_id
/// @return integer containing the relative minion id, values range from 0 to 1024
static inline int get_minion_from_thread(int relative_thread_id) {
  et_assert(relative_thread_id >= 0 && "Invalid thread Id (relative_thread_id < 0)");
  return relative_thread_id >> (device_config::config.threadsPerCore - 1);
}

/// @brief Obtains the relative minion id where the relative thread id is located
/// @param relative_thread_id
/// @return integer containing the relative minion id, values range from 0 to 1024
static inline int get_shire_from_minion(int minion_id) {
  et_assert(minion_id >= 0 && "Invalid minion Id (minion_id < 0)");
  // 32 minions per shire
  return minion_id >> 5;
}

static inline int get_num_entrypoints() {
    return (device_config::config.sameEntryPoint || device_config::config.threadsPerCore == 1) ? 1 : 2;
}

/**
 * \brief Namespace including low-level routines for the minion.
 **/
namespace hart {

/*! \cond PRIVATE */


/**
 * \brief Obtains the physical minion where the \p relative_thread_id is assigned.
 *
 * \param relative_thread_id Thread Id between 0 and get_num_threads()-1.
 * \return returns an integer with value between 0 and 1023
 */
static inline int get_physical_minion_id(int relative_thread_id) {
  // get the virtual minionId of the thread based on the threads per core configuration
  const auto relativeMinionId = get_minion_from_thread(relative_thread_id); 
  // offset is the physical id of the first  minion assigned to the current kernel
  const auto offset = __builtin_ctzll(device_config::env_->shire_mask) * SOC_MINIONS_PER_SHIRE;
  // return physical minion id
  return offset + relativeMinionId;
}

/**
 * \brief Computes integer log2(value).
 *
 * This function computes the base 2 logarithm of value and returns the result.
 *
 * \param value The number to apply the logarithm
 * \return The base 2 logarithm of value, floored to the closest integer.
 */
static constexpr size_t log2(size_t value) {
  size_t result = 0;
  while (value >>= 1) {
    result++;
  }
  return result;
}
/*! \endcond */

/** @enum Scope
 *  @brief list of possible Scope types in a barrier().
 */
enum class Scope {
  // future: kernel scope for multi-device support
  device, /*!< Synchronizes threads in the device (group by thread_id) */
  shire,  /*!< Synchronizes threads in a shire (group by thread_id) */
  minion  /*!< Synchronizes both thread_id=0 and thread_id=1 in a minion */
};

/**
 * \brief Synchronizes both threads in a minion
 *
 * Blocks thread execution until both threads in the minion reach the barrier.
 */
template <Scope S> inline typename std::enable_if_t<(S == Scope::minion), void> barrier() {
  constexpr uint32_t fcc = 1;                // Credit counter 0
  const uint32_t thread = get_hart_id() & 0x1; // Thread 0 or 1
  const auto localId = get_minion_id() % SOC_MINIONS_PER_SHIRE;
  const size_t hartMask = 1 << localId;

  // Note: barrier<shire> assumes full availability of FLB's for its synchronizations
  // Therefore we cannot make use of FLB's to implement barrier<minion> as they might be in use by the first.
  // We implement a 'ring-synchronization' method

  if (thread == 1) {
    fcc_consume(fcc);
    constexpr size_t destThread = 0;
    fcc_send(SHIRE_OWN, destThread, fcc, hartMask);
  } else {
    et_assert(thread == 0);
    constexpr size_t destThread = 1;
    fcc_send(SHIRE_OWN, destThread, fcc, hartMask);
    fcc_consume(fcc);
  }
}

/**
 * \brief Synchronizes threads per shire.
 *
 * Blocks thread execution until all selected harts in the shire reach the barrier. 
 * Threads in a shire that are synced can progress even if others shires still have not.
 * Barrier is only applied to threads executing the same entry point code.
 *
 * Every bit in \p hartMask maps to a hart in the shire.
 * 
 * Multiple barriers synchronizing different groups of threads per shire are supported. A thread
 * CANNOT be in more than one group at once.
 *
 * \param hartMask The n-th bit of the mask enables the sync of the n-th thread in the shire.
 * hartMask.count() must be a power of 2.
 * hartMask.count() max value is 32.
 */
template <Scope S>
inline typename std::enable_if_t<(S == Scope::shire), void>
barrier(std::bitset<64> hartMask = std::bitset<64>(0xFFFFFFFF)) {
  if (hartMask.count() < 2)
    return; // no threads to sync

  // Use credit counter 0
  constexpr uint32_t fcc = 0;

  // Get the local thread id (values: 0 or 1)
  const uint32_t localThread = get_hart_id() % 2;

  // Compute number of threads to sync in the FLB
  const auto numEntryPoints = get_num_entrypoints();
  const auto numThreadsToSync = static_cast<uint32_t>((device_config::config.threadsPerCore / numEntryPoints) * hartMask.count()) - 1U;

  // Each minion computes its corresponding FLB (values=[0,31])
  size_t flb = (get_minion_id() >> log2(hartMask.count())) % SOC_MINIONS_PER_SHIRE;
  
  // If odd and even harts have to sync separatedly
  if (localThread == 1 && numEntryPoints == 2) {
    flb += 16; // Shire minion Thread 1's use flbs [16, 31]
  }

  // Last minion to reach the barrier wakes up the others
  if (flbarrier(flb, numThreadsToSync)) {
    fcc_send(SHIRE_OWN, localThread, fcc, hartMask.to_ullong());
  }
  fcc_consume(fcc);
}

/*! \cond PRIVATE */
/**
 * \brief Synchronizes a range of threads in a device.
 *
 * Blocks thread execution until all the threads in the specified range reach this barrier. This barrier is shared only
 * between threads with the same thread_id (get_thread_id()).
 *
 * Synchronizes \p count contiguous threads starting at \p startingMinion.
 *
 * \param startingMinion First relative thread id in the synchronization range. Must be a multiple of 32 or 0.
 * \param count Number of threads in the synchronization range. Must be a multiple of 32.
 *
 */
template <Scope S>
inline typename std::enable_if_t<(S == Scope::device), void> barrier(size_t startingThread, size_t count) {
  constexpr uint32_t fcc = 1;
  constexpr std::bitset<64> allOnesMask = std::bitset<64>(0xFFFFFFFF);

  const auto thread = static_cast<uint32_t>(get_hart_id() % 2);
  const auto startingMinion = get_physical_minion_id(static_cast<int>(startingThread));
  const auto masterShireId = get_shire_from_minion(startingMinion);
  const auto numShires = static_cast<int>(count / (SOC_MINIONS_PER_SHIRE * device_config::config.threadsPerCore));

  et_assert((startingMinion % SOC_MINIONS_PER_SHIRE == 0) && "barrier: startingMinion must be multiple of 32");
  et_assert((count % SOC_MINIONS_PER_SHIRE == 0) && "barrier: count must be multiple of 32");
  et_assert((numShires < 33 && masterShireId < 32) && "barrier: shire configuration is INCORRECT");

  const auto localId = static_cast<int>(get_minion_id() % SOC_MINIONS_PER_SHIRE);
  const auto shireId = get_shire_from_minion(get_minion_id());
  const auto numEntryPoints = get_num_entrypoints();

  hart::barrier<Scope::shire>();

  if (masterShireId == shireId) {
    if (localId > masterShireId && localId < (masterShireId + numShires)) {
      const uint64_t flb = device_config::config.sameEntryPoint ? 0 : thread;
      const uint64_t numThreadsToSync = ((numShires - 1) * (numEntryPoints / device_config::config.threadsPerCore)) - 1;

      // wait for wake-up credit from each active shire
      fcc_consume(fcc);
      if (flbarrier(flb, numThreadsToSync)) {
        // wake-up all minion in master shire
        if constexpr (device_config::config.sameEntryPoint) {
          fcc_send(masterShireId, 0, fcc, allOnesMask.to_ullong());
          fcc_send(masterShireId, 1, fcc, allOnesMask.to_ullong());
        } else {
          fcc_send(masterShireId, thread, fcc, allOnesMask.to_ullong());
        }
      }
    }
    fcc_consume(fcc);
    // Invariant: at this point all shires are successfully synchronized

    // masterShire wakes up the other shires in the barrier
    if (localId > masterShireId && localId < (masterShireId + numShires)) {
      // Minion[i] wakes up shire[i], where i is the id of each assigned shire
      fcc_send(localId, thread, fcc, allOnesMask.to_ullong());
    }
  } else if (shireId < (masterShireId + numShires)) {
    // Global sync
    // Each active shire sends a credit to notify masterShire it has reached the barrier.
    if (localId == 0) {
      const uint64_t masterMinion = 1UL << shireId;
      fcc_send(masterShireId, thread, fcc, masterMinion);
    }
    // wait for wake-up from master shire
    fcc_consume(fcc);
  }
}
/*! \endcond */

/**
 * \brief Synchronizes a range of threads in a device.
 *
 * Blocks thread execution until all the threads in the specified range reach this barrier. 
 * Synchronizes \p count contiguous threads starting at \p startingThread.
 * \param startingThread First thread in the synchronization group. Must be a multiple of \p count or 0.
 * \param count Number of threads to synchronize. Must be a multiple of 32 or a power of two <= 32.
 */
inline void barrier(const size_t startingThread, const size_t count) {
  // Two count groups of values are supported.
  // If count >= 32. count must be a multiple of 32
  // If count < 32. count must be a power of two. i.e: {1,2,4,8,16}.
  std::bitset<64> cmask = std::bitset<64>(count);
  std::bitset<64> smask = std::bitset<64>(startingThread);

  // TODO: et_assert invalid range (start, count) configurations
  const auto numMinions = count / device_config::config.threadsPerCore;

  if (numMinions > SOC_MINIONS_PER_SHIRE) {
    barrier<Scope::device>(startingThread, count);
  } else {
    // check if count and startingMinion are powers of two (or 0)
    const bool isPow2 = (cmask.count() == 1) && ((smask.count() == 1) || (smask.count() == 0));
    et_assert(isPow2);
    const auto startingMinionId = get_minion_from_thread(static_cast<int>(startingThread));
    // size_t localId = startingMinionId % SOC_MINIONS_PER_SHIRE;
    const size_t localId = startingMinionId & 0x1F;
    // set mask bits corresponding to the range of minions to sync
    std::bitset<64> minionMask;
    for (size_t i = localId; i < localId + numMinions; i++) {
      minionMask.set(i);
    }
    barrier<Scope::shire>(minionMask);
  }
}

/*! \cond PRIVATE */
/**
 * \brief Synchronizes all threads in the kernel.
 *
 * Blocks thread execution until all the threads (with the same thread_id) reach this barrier.
 */
static inline void barrier() {
  const auto numShires = __builtin_popcountll(device_config::env_->shire_mask);
  const auto numThreads = get_num_threads();

  if (numShires > 1) {
    barrier<Scope::device>(0, numThreads);
  } else {
    barrier<Scope::shire>();
  }
} 
/*! \endcond */
} // namespace hart
#endif // GPSDK_SYNC_H
