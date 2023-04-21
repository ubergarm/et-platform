/*-------------------------------------------------------------------------
 * Copyright (C) 2018, Esperanto Technologies Inc.
 *
 ************************************************************
 * ---------------------------------------------------------
 * This code is Auto-Generated. Please DON'T MODIFY IT.
 * ---------------------------------------------------------
 ************************************************************
 *
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _SYNC_COMPUTE_NODE_H_
#define _SYNC_COMPUTE_NODE_H_

#if MINION_LOGGING
#include "logging.h"
#endif

#include "CommonCode.h"
#include "common_arguments.h"

// Number of credits to be sent to the act pref
#define N_CREDITS_TO_ACT_PREF 5
#define N_CREDITS_TO_WEIGHT_PREF 4

// Convert a numeric define to a string.
// It is necessary to use the 2 levels of conversion in the preprocessor
#define STRINGIFY(x) #x
#define TOSTRING(s) STRINGIFY(s)

void SyncComputeNode(uint32_t minionId, uint32_t shireId, uint32_t syncShireId, bool flushCB, bool flushL2,
                     [[maybe_unused]] uint64_t logHartHeaderAddress = 0);

static inline void sendInitCreditsToActPref(uint32_t minionId, uint32_t shireId,
                                            [[maybe_unused]] uint64_t logHartHeaderAddress = 0) {
  if (minionId < N_CREDITS_TO_ACT_PREF) {
    fcc_send(shireId, THREAD_1, FCC_0, HELPER_ACTIVATION_THREADS);
  }

  // one minion gives 1 credit to instruction prefetcher
  if (minionId == N_CREDITS_TO_ACT_PREF) {
    fcc_send(shireId, THREAD_1, FCC_1, HELPER_CODE_THREADS);
  }
}

static inline void sendInitCreditsToWeightPref(uint32_t minionId, uint32_t shireId,
                                               [[maybe_unused]] uint64_t logHartHeaderAddress = 0) {
  if ((minionId > N_CREDITS_TO_ACT_PREF) and (minionId <= (N_CREDITS_TO_ACT_PREF + N_CREDITS_TO_WEIGHT_PREF))) {
    fcc_send(shireId, THREAD_1, FCC_0, HELPER_WEIGHTS_THREADS);
  }
}

static inline void actPrefWaitsSyncCredit([[maybe_unused]] uint64_t logHartHeaderAddress = 0) {
  fcc(FCC_1);
}

static inline void drain_cbs_no_broad() {
  // Waits for all the banks to be idle
  for (uint32_t i = 0; i < L2_CACHE_BANKS; i++) {
    volatile uint64_t* sc_idx_cop_sm_ctl_addr_base = (uint64_t*)ESR_CACHE(SHIRE_OWN, i, SC_IDX_COP_SM_CTL_USER);

    uint64_t state;

    do {
      state = (*sc_idx_cop_sm_ctl_addr_base >> 24) & 0xFF;
    } while (state != 4);
  }

  // Starts the CB drain
  for (uint32_t i = 0; i < L2_CACHE_BANKS; i++) {
    volatile uint64_t* sc_idx_cop_sm_ctl_addr_base = (uint64_t*)ESR_CACHE(SHIRE_OWN, i, SC_IDX_COP_SM_CTL_USER);

    *sc_idx_cop_sm_ctl_addr_base = (1 << 0) | // Go bit = 1
                                   (10 << 8); // Opcode = CB_Inv (Coalescing buffer invalidate)
  }

  // Checks done
  for (uint32_t i = 0; i < L2_CACHE_BANKS; i++) {
    volatile uint64_t* sc_idx_cop_sm_ctl_addr_base = (uint64_t*)ESR_CACHE(SHIRE_OWN, i, SC_IDX_COP_SM_CTL_USER);

    uint64_t state;

    do {
      state = (*sc_idx_cop_sm_ctl_addr_base >> 24) & 0xFF;
    } while (state != 4);
  }
}

void CBDrainersCode(uint32_t minionId, uint32_t shireId, uint32_t nNodes,
                    [[maybe_unused]] uint64_t logHartHeaderAddress = 0);

#endif //_SYNC_COMPUTE_NODE_H_
