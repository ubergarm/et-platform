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

#ifndef UBER_KERNEL_COMMON_H
#define UBER_KERNEL_COMMON_H

#include <inttypes.h>

// Id of the first synchronization minion in the master/sync shire
static constexpr int FIRST_SYNC_MINION = 16;

// Helper thread masks
static auto constexpr COMPUTE_THREADS = 0xFFFFFFFFULL;
static auto constexpr HELPER_ACTIVATION_THREADS = 0x0000FFFFULL;
static auto constexpr HELPER_WEIGHTS_THREADS = 0x0F0F0000ULL;
static auto constexpr HELPER_DRAIN_THREADS = 0x30300000ULL;
static auto constexpr HELPER_CODE_THREADS = 0x40000000ULL;
static auto constexpr HELPER_DDR_THREADS = 0x80000000ULL;
static auto constexpr HELPER_EVICT_W_THREADS = 0x80000000ULL;
static auto constexpr SYNC_MINIONS = 0xFFFF0000ULL;

// Get the mask of threads that are sync threads in the sync shire
static inline uint64_t getSyncThreadsMask(int activeShires) {
  return ((1ULL << activeShires) - 1ULL) << (FIRST_SYNC_MINION * 2);
}

#endif // UBER_KERNEL_COMMON_H
