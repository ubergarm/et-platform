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

#include <type_traits>
#include <bitset>

// Device common
#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

// FW syscall IDs
#include <etsoc/isa/syscall.h>

#include <profiling.h>

#include "environment.h"
#include "entryPoint.h"

extern Arguments * args_;
extern DeviceConfig config;

/**
 * \brief Namespace including low-level routines for the minion.
 **/
namespace hart {

/*! \cond PRIVATE */
/**
 * \brief Computes integer log2(value).
 *
 * This function computes the base 2 logarithm of value and returns the result.
 *
 * \param value The number to apply the logarithm
 * \return The base 2 logarithm of value, floored to the closest integer.
 */
constexpr size_t log2(size_t value) {
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
 * \brief Synchronizes both threads per minion
 *
 * Blocks thread execution until both threads in the minion reach the barrier.
 */
template <Scope S> inline typename std::enable_if_t<(S == Scope::minion), void> barrier() {
  constexpr uint32_t fcc = 1;                // Credit counter 0
  const uint32_t thread = get_hart_id() & 1; // Thread 0 or 1
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
 * Blocks thread execution until all threads specified in the \p hartMask reach the barrier. This barrier is shared only
 * between threads with the same thread_id (get_thread_id()).
 *
 * Every bit of hartMask maps to a hart in the shire.
 *
 * Multiple barriers synchronizing different groups of threads per shire are supported, as long as there is not a thread
 * in more than one group at once.
 *
 * \param hartMask The n-th bit of the mask enables the sync of the n-th thread in the shire.
 * hartMask.count() must be a power of 2.
 * hartMask.count() max value is 32.
 *
 */
// template <Scope S>
// inline typename std::enable_if_t<(S == Scope::shire), void>
// barrier(std::bitset<64> hartMask = std::bitset<64>(0xFFFFFFFF)) {
//   if (hartMask.count() < 2)
//     return; // no threads to sync

//   constexpr uint32_t fcc = 0;                // Credit counter 0
//   const uint32_t thread = get_hart_id() % 2; // Thread 0 or 1

//   // Each minion computes its corresponding FLB (values=[0,31])
//   size_t flb = (get_minion_id() >> log2(hartMask.count())) % SOC_MINIONS_PER_SHIRE;
//   if (thread == 1) {
//     flb += 16; // Shire minion Thread 1's use flbs [16,31]
//   }
//   // Last minion to reach the barrier wakes up the others
//   if (flbarrier(flb, hartMask.count() - 1)) {
//     fcc_send(SHIRE_OWN, thread, fcc, hartMask.to_ullong());
//   }
//   fcc_consume(fcc);
// }

/// @brief  TEST
/// @tparam S 
/// @tparam threadsPerCore 
/// @tparam numEntryPoints 
/// @param hartMask 
/// @return 
template <Scope S>
inline typename std::enable_if_t<(S == Scope::shire), void>
barrier(std::bitset<64> hartMask = std::bitset<64>(0xFFFFFFFF), int threadsPerCore = 1, int numEntryPoints = 1) {
  // TODO; Explore the use of inline constexpr with threadsPerCore and EntryPoint as template parameters to reduce branches in the code.
  // https://stackoverflow.com/questions/30208685/how-to-declare-constexpr-extern
  if (hartMask.count() < 2)
    return; // no threads to sync

  constexpr uint32_t fcc = 0;                     // Credit counter 0
  const uint32_t localThread = get_hart_id() % 2; // Thread 0 or 1

  // Get the number of threads to sync under a FLB
  const uint32_t numThreadsToSync = static_cast<uint32_t>((threadsPerCore / numEntryPoints) * hartMask.count());

  // Each minion computes its corresponding FLB (values=[0,31])
  size_t flb = (get_minion_id() >> log2(hartMask.count())) % SOC_MINIONS_PER_SHIRE;

  // If odd and even harts have to sync separatedly
  if (localThread == 1 && numEntryPoints == 2) {
    flb += 16; // Shire minion Thread 1's use flbs [16, 31]
  }

  // Last minion to reach the barrier wakes up the others
  if (flbarrier(flb, numThreadsToSync - 1)) {
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
  constexpr uint32_t fcc = 1;               // Credit counter 0
  constexpr uint32_t thread_0 = 0;
  constexpr uint32_t thread_1 = 1;
  constexpr std::bitset<64> allOnesMask = std::bitset<64>(0xFFFFFFFF);

  const bool sameEntryPoint = (config.entryPoint_0 == config.entryPoint_1);
  const uint32_t thread = get_hart_id() % 2;
  const auto startingRelativeMinionId = startingThread / config.threadsPerCore; // (virtual) minionId of the starting kernel thread
  const auto startingMinion = (static_cast<int>(__builtin_ctzll(args_->env.shireMask)) * SOC_MINIONS_PER_SHIRE) + startingRelativeMinionId; // global starting minion id
  const auto masterShireId = (uint32_t) startingMinion / SOC_MINIONS_PER_SHIRE;  // ID of the 'master shire' that coordinates the sync
  const size_t numShires = count / (SOC_MINIONS_PER_SHIRE * config.threadsPerCore);   // number of total shires to sync

  et_assert((startingMinion % SOC_MINIONS_PER_SHIRE == 0)  && "barrier: startingMinion must be multiple of 32");
  et_assert((count % SOC_MINIONS_PER_SHIRE == 0) && "barrier: count must be multiple of 32");
  et_assert((numShires < 33 && masterShireId < 32) && "barrier: shire configuration is INCORRECT");

  const auto localId = get_minion_id() % SOC_MINIONS_PER_SHIRE;
  const auto shireId = get_minion_id() / SOC_MINIONS_PER_SHIRE;
  const int numEntryPoints = (config.entryPoint_0 == config.entryPoint_1 || config.threadsPerCore == 1) ? 1 : 2;

  hart::barrier<Scope::shire>(std::bitset<64>(0xFFFFFFFF), config.threadsPerCore, numEntryPoints);

  if (masterShireId == shireId) {
    if (localId > masterShireId && localId < (masterShireId + numShires)) {
      const uint64_t flb = sameEntryPoint ? 0 : thread;
      const uint64_t flbCount = ((numShires - 1) * (numEntryPoints / config.threadsPerCore)) - 1;

      // wait for wake-up credit from each active shire
      fcc_consume(fcc);
      if (flbarrier(flb, flbCount)) {
        // wake-up all minion in master shire
        if (sameEntryPoint) {
          fcc_send(masterShireId, thread_0, fcc, allOnesMask.to_ullong());
          fcc_send(masterShireId, thread_1, fcc, allOnesMask.to_ullong());
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
      et_printf("I am shire: %u, minion: %u, thread: %u, sending a credit to master. \n", shireId, localId, thread);
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
 * Blocks thread execution until all the threads in the specified range reach this barrier. This barrier is shared only
 * between threads with the same thread_id (get_thread_id()). Synchronizes \p count contiguous threads starting at \p
 * startingMinion.
 * \param startingMinion First minion in the synchronization group. Must be a multiple of \p count or 0.
 * \param count Number of threads to synchronize. Must be a multiple of 32 or a power of two <= 32.
 */
// inline void barrier(const size_t startingMinion, const size_t count) {
//   // Two count groups of values are supported.
//   // If count >= 32. count must be a multiple of 32
//   // If count < 32. count must be a power of two. i.e: {1,2,4,8,16}.
//   std::bitset<64> cmask = std::bitset<64>(count);
//   std::bitset<64> smask = std::bitset<64>(startingMinion);

//   // check if count and startingMinion are powers of two (or 0)
//   bool isPow2 = (cmask.count() == 1) && ((smask.count() == 1) || (smask.count() == 0));
//   // TODO: et_assert invalid range (start, count) configurations

//   if (count > SOC_MINIONS_PER_SHIRE) {
//     barrier<Scope::device>(startingMinion, count);
//   } else {
//     et_assert(isPow2);
//     // get the startingMinion id inside the shire
//     size_t localId = startingMinion % SOC_MINIONS_PER_SHIRE;
//     std::bitset<64> mask;
//     // set mask bits fromt local id to
//     for (size_t i = localId; i < localId + count; i++) {
//       mask.set(i);
//     }
//     barrier<Scope::shire>(mask);
//   }
// }

inline void barrier(const size_t startingThread, const size_t count) {
  // Two count groups of values are supported.
  // If count >= 32. count must be a multiple of 32
  // If count < 32. count must be a power of two. i.e: {1,2,4,8,16}.
  std::bitset<64> cmask = std::bitset<64>(count);
  std::bitset<64> smask = std::bitset<64>(startingThread);

  // check if count and startingMinion are powers of two (or 0)
  bool isPow2 = (cmask.count() == 1) && ((smask.count() == 1) || (smask.count() == 0));
  // TODO: et_assert invalid range (start, count) configurations
  const auto numMinions = count / config.threadsPerCore;
  
  if (numMinions > SOC_MINIONS_PER_SHIRE) {
    barrier<Scope::device>(startingThread, count);
  } else {
    et_assert(isPow2);
    const auto startingMinionId = startingThread / config.threadsPerCore; // replace by a shift
    size_t localId = startingMinionId % SOC_MINIONS_PER_SHIRE;

    // set mask bits fromt local id to
    std::bitset<64> minionMask;
    for (size_t i = localId; i < localId + numMinions; i++) {
      minionMask.set(i);
    }
    const int numEntryPoints = (config.entryPoint_0 == config.entryPoint_1 || config.threadsPerCore == 1) ? 1 : 2;
    barrier<Scope::shire>(minionMask, config.threadsPerCore, numEntryPoints);
  }
}


/*! \cond PRIVATE */
/**
 * \brief Synchronizes all threads in the kernel.
 *
 * Blocks thread execution until all the threads (with the same thread_id) reach this barrier.
 * TEMPORARY: not working if the assigned minions to the kernel is different than 1024
 * use the overloaded function barrier(startingMinon, count) instead.
 */
inline void barrier() {
  const auto numThreads = get_num_threads();
  
  if ((numThreads / config.threadsPerCore) > SOC_MINIONS_PER_SHIRE) {
    barrier<Scope::device>(0, numThreads);
  } else {
    int numEntryPoints = (config.entryPoint_0 == config.entryPoint_1 || config.threadsPerCore == 1 ) ? 1 : 2;
    barrier<Scope::shire>(std::bitset<64>(0xFFFFFFFF), config.threadsPerCore, numEntryPoints);
  }
}
/*! \endcond */
}
#endif // GPSDK_SYNC_H
