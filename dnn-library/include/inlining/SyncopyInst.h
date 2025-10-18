/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
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

#include "LibTensor.h"
#include "utils.h"

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
INLINE_ATTR void
fwdLibSyncopyInst(LibTensor* outT, LibTensor* inT, unsigned int off, [[maybe_unused]] const uint64_t flags,
                  [[maybe_unused]] const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  size_t hart = get_hart_id();
  size_t threadId = hart & 1;
  size_t minionId = (hart >> 1) - (32 * 32 + 16);
  size_t activeMinions = (assignedMinions == 0) ? 16 : assignedMinions;
  // Disable second thread from now on, as they don't have tensor extensions
  if ((threadId != 0) || (minionId >= activeMinions)) {
    return;
  }

  void* dst = outT->getRawDataPointer();
  void* src = inT->getRawDataPointer();

  // Need to use the sizes of the output tensor. Graph might change them slightly (vs the input dimensions) to make
  // sure the function doesn't go beyond the tensor limits (can happen when the source tensor is a son with an offset
  // with respect to the parent)
  const dim_t* index = outT->dims().data();
  const dim_t* pitch = outT->strides().data();

  size_t typeSize = getsize<srcType>();
  size_t numBytes = pitch[0] * index[0] * typeSize + off;       // Total number of elements in the tensor
  size_t numCacheLines = (numBytes - 1) / CACHE_LINE_BYTES + 1; // 64 = CacheLineLength
  int64_t  minionCacheLines = (numCacheLines - 1) / activeMinions + 1;
  size_t initialCacheLine = minionCacheLines * minionId;
  size_t lastCacheLine = initialCacheLine + minionCacheLines;
  minionCacheLines = (lastCacheLine <= numCacheLines)   ? minionCacheLines
                   : (initialCacheLine < numCacheLines) ? numCacheLines - initialCacheLine
                   :                                      0;

  // Computes source and destination
  size_t srcAddr = ((size_t)src & ~0x3FUL) + initialCacheLine * CACHE_LINE_BYTES; // Aligns to cache line
  size_t dstAddr = (size_t)dst + initialCacheLine * CACHE_LINE_BYTES;
  size_t srcAddrOrig = srcAddr;

  // Saves minion cache lines for evicts
  size_t minionCacheLinesOrig = minionCacheLines;

  // Actual copies doing tensorload + tensorstore
  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
  while (minionCacheLines > 0) {
    // Caps to 16 cachelines
    size_t cl = (minionCacheLines >= 16) ? 0xFUL : minionCacheLines - 1;
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
    size_t pending = 4 - activeMinions;
    if (minionId < pending) {
      cache_ops_cb_drain(SHIRE_OWN, minionId + activeMinions);
    }
  }

  // Evicts read data from L2 if not L2scp
  if ((srcAddrOrig >> 31) != 1U) { // TODO : get parameter from neuralizer and use constexpr
    evict_va_multi(uint64_t(cop_dest::to_L3), srcAddrOrig, minionCacheLinesOrig);
  }

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
