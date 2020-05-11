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
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "Float16.h"
#include "Writer.h" // From include/internal path
#include "Addresser.h" // From include/internal path
#include "Converter.h" // From include/internal path
#include "Operator.h" // From include/internal path
#include "utils.h" // From include/internal path
#include "shire.h"
#include "barriers.h"
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Sync Minions copy the src matrix to the dst matrix
 *
 * It makes a copy of the tensor src into the dst tensor, which will
 * have the same pitches and dimensions. An offset was added to enforce
 * tensors starting at cacheline.
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
template <typename srcType>
inline void fwdLibSyncopyInstTensorized(LibTensor* outT, LibTensor* inT,
                                        unsigned int off) {

  uint32_t hart = get_hart_id();
  uint32_t threadId = hart & 1;
  uint32_t minionId = (hart >> 1) - (32 * 32 + 16);
  int32_t activeMinions = 16;
  // Disable second thread from now on, as they don't have tensor extensions
  if (threadId != 0) { return; }

  
  /* maintain compatibility through the new Iface Libtensor */

  void *dst = outT->getRawDataPointer<void>();
  void *src = inT->getRawDataPointer<void>();
  
  // unsigned int *Index = (unsigned int *) Dims;
  const dim_t *Index = inT->dims().data();
  // unsigned int *Pitch = (unsigned int *) Pitches;
  const dim_t *Pitch = inT->strides().data();
  
  size_t typeSize = getsize<srcType>();
  uint64_t numBytes = Pitch[0] * Index[0] * typeSize + off; // Total number of elements in the tensor
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
  uint64_t dstAddrOrig = dstAddr;

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

  // Barrier to get all minions threa0 here
  // Guarantees all data in CB
  shire_barrier(0,                 // FLB0
                0,                 // FCC0
                activeMinions,     // Number of active minions
                SYNC_MINIONS_MASK, // Mask of active thread0 minions
                0);                // Mask of active thread1 minions

  // Flush CB (4 banks, 4 minions in parallel)
  if(minionId < 4)
    cb_drain(SHIRE_OWN, minionId);

  // Barrier to get all minions threa0 here
  // Guarantees all data in L3
  shire_barrier(0,                 // FLB0
                0,                 // FCC0
                activeMinions,     // Number of active minions
                SYNC_MINIONS_MASK, // Mask of active thread0 minions
                0);                // Mask of active thread1 minions

  // Evicts from L3 to DDR
  minionCacheLines = minionCacheLinesOrig;
  dstAddr = dstAddrOrig;
  while (minionCacheLines > 0) {
    // Caps to 16 cachelines
    uint32_t cl = (minionCacheLines >= 16) ? 0xF : minionCacheLines-1;
    // Flush L3 to DDR
    evict_va_multi(0x3, dstAddr, cl);
    // Update for next iteration
    minionCacheLines -= 16;
    dstAddr += 1024;
  }
  WAIT_CACHEOPS;
}

} // namespace dnn_lib

} // namespace inlining

#endif // _SYNCOPY_INST_TENSORIZED_H_
