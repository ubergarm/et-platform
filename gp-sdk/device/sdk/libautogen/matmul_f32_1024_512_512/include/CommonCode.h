/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *
 ************************************************************
 * ---------------------------------------------------------
 * This code is Auto-Generated. Please DON'T MODIFY IT.
 * ---------------------------------------------------------
 ************************************************************
 *-------------------------------------------------------------------------
 */

#ifndef _COMMON_CODE_H_
#define _COMMON_CODE_H_

// Global
#include <inttypes.h>

// Device common
#include <etsoc/isa/utils.h>
#include <etsoc/common/utils.h>


// FW syscall IDs
#include <etsoc/isa/syscall.h>
#include <stdio.h>

#include "UberKernelCommon.h"

// Some defines
#define THREAD_0 0
#define THREAD_1 1
#define FCC_0 0
#define FCC_1 1

#define SC_CACHEOP_L2_EVICT 0x3

// Global functions

// This function sends one FCC to a sync minion when the last hart gets
// to the barrier
static inline void global_barrier_starter(uint64_t num_harts,    // Harts doing barrier in the source shires
                                          uint64_t flb_num,      // FLB to be used in the source shire for the barrier
                                          uint64_t shire_id_src, // Source shire id
                                          uint64_t shire_id_dst, // Source shire id
                                          uint64_t fcc)          // Which FCC to send the shire ready signal
{
  volatile uint64_t* sync_minion_addr = (uint64_t*)((1ULL << 32)                // ESR
                                                    + (shire_id_dst << 22)      // Going to master shire
                                                    + (0x1AULL << 17)           // Shire other ESRs
                                                    + 0xC0ULL                   // FCC ESRs
                                                    + ((shire_id_src & 1) * 16) // Which thread is going to
                                                    + (fcc * 8));               // FCC destination
  uint64_t sync_minion_data = 1ULL << ((shire_id_src / 2) + 16);                // Send FCC to according sync minion
  uint64_t flb_result = flb(flb_num, num_harts - 1);
  if (flb_result == 1) {
    *sync_minion_addr = sync_minion_data;
  }
}

// This function waits for one FCC to a sync minion when the last hart gets
// to the barrier
static inline void
global_barrier_receiver(uint64_t fcc_wait,           // Which FCC to wait for
                        uint64_t flb_num,            // FLB to be used in the source shire for the barrier
                        uint64_t minion_id,          // Sync minion id
                        uint64_t thread_id,          // Sync thread id
                        uint64_t thread_dest,        // Thread of the FCC dest
                        uint64_t fcc_dest,           // FCC for dest
                        uint64_t minion_mask_dest,   // Mask of minions in dest shire to receive FCC
                        uint64_t n_compute_shires,   // number of compute Shires
                        uint64_t sync_thread_0_mask, // mask of threads 0 that will receive credits
                        uint64_t sync_thread_1_mask, // mask of threads 1 that will receive credits
                        uint64_t sync_shire_id,      // Where the sync minions are placed
                        bool print,
                        [[maybe_unused]] uint64_t logHartHeaderAddress = 0) { // logging hart header address
  // If n_compute_shires is 0, means that the barrier is not desired
  if (n_compute_shires > 0) {
    // Waits for its associated shire FCC and do FLB
    fcc(fcc_wait);
    uint64_t flb_result = flb(flb_num, n_compute_shires - 1);
    if (print) {
      et_printf("  shire %i done\n", (int)((minion_id << 1) + thread_id) - 32);
    }

    // If last wake up other sync minions
    if (flb_result == 1) {
      // Send to FCC1
      fcc_send(sync_shire_id, THREAD_0, FCC_1, sync_thread_0_mask);
      fcc_send(sync_shire_id, THREAD_1, FCC_1, sync_thread_1_mask);
    }

    // Waits for FCC that last sync minion got FCC
    fcc(FCC_1);
  }

  // Sends FCC to destination, if minion_mask_dest is empty, no credit sent
  if (minion_mask_dest != 0) {
    fcc_send(((minion_id - 16) * 2) + thread_id, thread_dest, fcc_dest, minion_mask_dest);
  }
}

static inline void ecall_shire_cache_bank_op(uint64_t shire, uint64_t bank, uint64_t op) {
  syscall(SYSCALL_SHIRE_CACHE_BANK_OP, shire, bank, op);
}

#endif
