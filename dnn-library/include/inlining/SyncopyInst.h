/*-------------------------------------------------------------------------
 * Copyright (C) 2019, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _SYNCOPY_INST_TENSORIZED_H_
#define _SYNCOPY_INST_TENSORIZED_H_

#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

#include <etsoc/isa/barriers.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include "utils.h" // From include/internal path
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Sync Minions copy the src matrix to the dst matrix
 *
 * It makes a copy of the tensor src into the dst tensor, which will
 * have the same pitches and dimensions. An offset was added to enforce
 * tensors starting at cacheline. The L1 and L2 after the node do not
 * keep any contents of the src or destination, so not eviction is required
 *
 * @tparam srcType The type of the elements in the tensor.
 * @param[out] dst Pointer to the output matrix.
 * @param[in] Dims The "number of dimensions" of the matrix.
 * @param[in] Pitches Vector of pitches of the matrix.
 * @param[in] src Pointer to the input matrix.
 * @param[in] DimNum The "number of dimensions" of the matrix.
 * @param[in] scale, offset Parameters for the quantization.
 * @param[in] off offset applied to ensure tensor starting at CL
 */

template <ElemKind elK>
INLINE_ATTR void fwdLibSyncopyInst(LibTensor* outT, LibTensor* inT, unsigned int off, const uint64_t flags,
                                   const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  uint32_t hart = get_hart_id();
  uint32_t threadId = hart & 1;
  uint32_t minionId = (hart >> 1) - (32 * 32 + 16);
  uint32_t activeMinions = (assignedMinions == 0) ? 16 : assignedMinions;
  // Disable second thread from now on, as they don't have tensor extensions
  if ((threadId != 0) || (minionId >= activeMinions)) {
    return;
  }

  void *dst = outT->getRawDataPointer<void>();
  void *src = inT->getRawDataPointer<void>();

  // Need to use the sizes of the output tensor. Graph might change them slightly (vs the input dimensions) to make
  // sure the function doesn't go beyond the tensor limits (can happen when the source tensor is a son with an offset
  // with respect to the parent)
  const dim_t* index = outT->dims().data();
  const dim_t* pitch = outT->strides().data();

  size_t typeSize = getsize<srcType>();
  uint64_t numBytes = pitch[0] * index[0] * typeSize + off;       // Total number of elements in the tensor
  uint64_t numCacheLines = (numBytes - 1) / CACHE_LINE_BYTES + 1; // 64 = CacheLineLength
  int64_t  minionCacheLines = (numCacheLines - 1) / activeMinions + 1;
  uint64_t initialCacheLine = minionCacheLines * minionId;
  uint64_t lastCacheLine = initialCacheLine + minionCacheLines;
  minionCacheLines = (lastCacheLine <= numCacheLines)   ? minionCacheLines
                   : (initialCacheLine < numCacheLines) ? numCacheLines - initialCacheLine
                   :                                      0;

  // Computes source and destination
  uint64_t srcAddr = ((uint64_t) src & ~0x3FULL) + initialCacheLine * CACHE_LINE_BYTES; // Aligns to cache line
  uint64_t dstAddr = (uint64_t) dst + initialCacheLine * CACHE_LINE_BYTES;
  uint64_t srcAddrOrig = srcAddr;

  // Saves minion cache lines for evicts
  int64_t minionCacheLinesOrig = minionCacheLines;

  // Actual copies doing tensorload + tensorstore
  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
  while (minionCacheLines > 0) {
    // Caps to 16 cachelines
    uint32_t cl = (minionCacheLines >= 16) ? 0xF : minionCacheLines-1;
    // Copies source to L1 Scp and makes sure they are available
    // for the tensor store
    tensor_load(0, 0, 0, 0, 0, srcAddr, 0, cl, 0x40, 0);
    WAIT_TENSOR_LOAD_0;
    // Need to wait for TStore to prevent overriding L1 scp next iteration
    // with the tensor loads
    tensor_store_scp(0, 0, cl, dstAddr, 0x40);
    WAIT_TENSOR_STORE;
    // Update for next iteration
    minionCacheLines -= 16;
    srcAddr += 1024;
    dstAddr += 1024;
  }

  // Barrier to get all minions thread0 here
  // Guarantees all data in CB
  shire_barrier(0,                 // FLB0
                0,                 // FCC0
                activeMinions,     // Number of active minions
                (((1 << activeMinions)-1) << 16),     // Mask of active thread0 minions
                0);                // Mask of active thread1 minions

  // Flush CB (4 banks)
  if (activeMinions >= 4) {
    // 4 minions in parallel
    if (minionId < 4) {
      cache_ops_cb_drain(SHIRE_OWN, minionId);
    }
  } else if (activeMinions == 1) {
    cache_ops_cb_drain(SHIRE_OWN, 0);
    cache_ops_cb_drain(SHIRE_OWN, 1);
    cache_ops_cb_drain(SHIRE_OWN, 2);
    cache_ops_cb_drain(SHIRE_OWN, 3);
  } else {
    cache_ops_cb_drain(SHIRE_OWN, minionId);
    uint32_t pending = 4 - activeMinions;
    if (minionId < pending) {
      cache_ops_cb_drain(SHIRE_OWN, minionId + activeMinions);
    }
  }

  // Evicts read data from L2
  evict_va_multi(0x2, srcAddrOrig, minionCacheLinesOrig);
  WAIT_CACHEOPS;

  // Barrier to get all minions thread0 here
  // Guarantees all data in L3
  shire_barrier(0,                 // FLB0
                0,                 // FCC0
                activeMinions,     // Number of active minions
                (((1 << activeMinions)-1) << 16),     // Mask of active thread0 minions
                0);                // Mask of active thread1 minions
}

} // namespace dnn_lib

} // namespace inlining

#endif // _SYNCOPY_INST_TENSORIZED_H_
