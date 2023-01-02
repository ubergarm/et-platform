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

namespace hart {

// temporary fix for not having a way to obtain assigned minions in the device;

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

/** @enum hart::Scope
 *  @brief enum class representing the thread scope synced in a barrier.
 */
enum class Scope {
  // future: kernel scope for multi-device support
  device, /**< Sync threads in the device  */
  shire,  /**< Sync threads in a shire */
  minion  /**< Sync threads per minion */
};

/**
 * \brief Sync barrier for threads within a shire.
 *
 * Blocks thread execution until all the threads selected in the hartMask reach this barrier.
 * Every bit of hartMask maps to a hart in the shire.
 * Multiple concurrent barriers per shire are supported, as long as there is no thread overlapping between them. 
 * 
 * \param hartMask The n-th bit of the mask enables the sync of the n-th thread in the shire.
 * hartMask.count() must be a power of 2.
 * hartMask.count() max value is 32.
 *  
 */
template <Scope S>
inline typename std::enable_if_t<(S == Scope::shire), void> 
barrier(std::bitset<64> hartMask = std::bitset<64>(0xFFFFFFFF)) {
  
  constexpr uint32_t fcc = 0;     // Credit counter 0
  constexpr uint32_t thread0 = 0; // Thread 0

  // Each minion computes its corresponding FLB
  size_t flb = get_minion_id() >> log2(hartMask.count());

  if (flbarrier(flb, hartMask.count() - 1)) {
    fcc_send(SHIRE_OWN, thread0, fcc, hartMask.to_ullong());
  }
  fcc_consume(fcc);
}

/**
 * \brief Sync barrier for a group of threads within a device.
 * 
 * Internal function, specific to handle barrier() when count > 32.
 * 
 * Blocks thread execution until all the threads in the synchronization group reach this barrier.
 * The synchronization group syncs count contiguous threads starting from 'startingMinion'.
 * 
 * \param startingMinion First thread in the synchronization group. Must be a multiple of 32
 * \param count Number of threads to synchronize. Must be a multiple of 32.
 * 
 */
template <Scope S>
inline typename std::enable_if_t<(S == Scope::device), void> 
barrier(size_t startingMinion, size_t count) {
  constexpr uint32_t fcc0 = 0;     // Credit counter 0
  constexpr uint32_t thread0 = 0;  // Thread 0
  const std::bitset<64> allOnesMask = std::bitset<64>(0xFFFFFFFF);
  const std::bitset<64> minion0Mask = std::bitset<64>(0x1);

  auto masterShireId = (uint32_t) startingMinion / SOC_MINIONS_PER_SHIRE; // Master shire that coordinates sync
  size_t numShires = count / SOC_MINIONS_PER_SHIRE;

  et_assert((startingMinion % SOC_MINIONS_PER_SHIRE == 0)  && "barrier: startingMinion must be multiple of 32");
  et_assert((count % SOC_MINIONS_PER_SHIRE == 0)  && "barrier: count must be multiple of 32");
  et_assert(numShires < 33 && masterShireId < 32 && "barrier: shire configuration is INCORRECT");

  auto localId = get_minion_id() % SOC_MINIONS_PER_SHIRE;
  auto shireId =  get_minion_id() / SOC_MINIONS_PER_SHIRE;

  // Local sync - syncs all minions in the same shire
  hart::barrier<Scope::shire>(allOnesMask);

  if (masterShireId == shireId) {
    // minion 0 coordinates syncing with the other shires
    if (localId == 0) {
      // minion 0 waits for (numShires - 1) credits
      // for loop starts at 1 because already consumed 1
      for (size_t i = 0; i < numShires - 1; i++) {
        fcc_consume(fcc0);
      }
      // time to wake up the other shires
      for (auto sId = masterShireId; sId < masterShireId + numShires; sId++) {
        fcc_send(sId, thread0, fcc0, allOnesMask.to_ullong());
      }
    }
  } else if (shireId < masterShireId + numShires) {
    // Non-Master minion sends a credit to wake up the Master Shire - minion 0 - thread 0.
    if (localId == 0) {
      fcc_send(masterShireId, thread0, fcc0, minion0Mask.to_ullong());
    }
  }

  // Everyone sleeps until the Master Shire wakes everyone including itself
  fcc_consume(fcc0);
}


/**
 * \brief Sync barrier for a group of threads within a device.
 *
 * Blocks thread execution until all the threads in the synchronization group reach this barrier.
 * The synchronization group syncs count contiguous threads starting from 'startingMinion'.
 * 
 * \param startingMinion First thread in the synchronization group. If count is > 32, it must be a multiple of 32, otherwise, it must be a power of two or 0.
 * \param count Number of threads to synchronize. If count is > 32, it must be a multiple of 32, otherwise, it must be a power of two.
 * 
 */
inline void barrier(const size_t startingMinion, const size_t count) {
  // Two count groups of values are supported.
  // If count > 32. count must be a multiple of 32. i.e: {64,96,128,160,..,1024}
  // If count <= 32. count must be a power of two. i.e: {1,2,4,8,16,32}.

  std::bitset<64> cmask = std::bitset<64>(count);
  std::bitset<64> smask = std::bitset<64>(startingMinion);

  // check if count and startingMinion are powers of two (or 0)
  bool isPow2 = (cmask.count() == 1) && ((smask.count() == 1) || (smask.count() == 0));

  if (count > SOC_MINIONS_PER_SHIRE) {
    barrier<Scope::device>(startingMinion, count);
  } else {
    et_assert(isPow2);
    size_t localId = get_minion_id() % SOC_MINIONS_PER_SHIRE;

    std::bitset<64> mask;
    for (size_t i = localId; i < localId + count ; i++) {
      mask.set(i);
    }
    barrier<Scope::shire>(mask);
  }
}

/**
 * \brief Sync barrier for all threads in the kernel.
 *
 * Blocks thread execution until all the threads reach this barrier.
 * TEMPORARY: not working if the assigned minions to the kernel is different than 1024
 * use the overloaded function barrier(startingMinon, count) instead.
 */
inline void barrier() {
  size_t assignedMinions = 1024;

  if (assignedMinions > SOC_MINIONS_PER_SHIRE) {
    barrier<Scope::device>(0, assignedMinions);
  } else {
    barrier<Scope::shire>(std::bitset<64>(0xFFFFFFFF));
  }
}

}
#endif // GPSDK_SYNC_H
