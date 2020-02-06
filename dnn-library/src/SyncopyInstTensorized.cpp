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

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "LibNodes.h"
#include "GenInstances.h"
#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"

using namespace std;

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
 void dnn_lib::fwdLibSyncopyInstTensorized(void *dst, void *Dims, void *Pitches,
                                           void *src, unsigned int DimNum,
                                           float *scale, int32_t *offset,
                                           unsigned int off) {

  int32_t minionId = get_minion_id() - (32 * 32 + 16);
  int32_t activeMinions = 16;
  if ((minionId < 0) || (minionId >= activeMinions))
    return;

  unsigned int *Index = (unsigned int *)Dims;
  unsigned int *Pitch = (unsigned int *)Pitches;

  size_t typeSize = getsize<srcType>();
  uint64_t numElems = Pitch[0] * Index[0] *
                             typeSize + off; // Total number of elements in the tensor
  uint64_t numCacheLines = (numElems - 1) / 64 + 1; // 64 = CacheLineLength
  uint64_t minionCacheLines = (numCacheLines - 1) / activeMinions + 1;
  uint64_t initialCacheLine = minionCacheLines * minionId;
  uint64_t lastCacheLine = initialCacheLine + minionCacheLines;
  minionCacheLines =
          (lastCacheLine <= numCacheLines) ? minionCacheLines
        : (initialCacheLine < numCacheLines) ? numCacheLines - initialCacheLine : 0;
  uint64_t srcAddr = (uint64_t)src + initialCacheLine*64;
  uint64_t dstAddr = (uint64_t)dst + initialCacheLine*64;

  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
  while (minionCacheLines > 0) {
    uint32_t cl = (minionCacheLines >= 16) ? 0xF : minionCacheLines-1;
    tensor_load(0, 0, 0, 0, 0, srcAddr, 0, cl, 0x40, 0);
    WAIT_TENSOR_LOAD_0;
    srcAddr += 1024;
    minionCacheLines -= 16;
    tensor_store_scp(0, 0, cl, dstAddr, 0x40);
    dstAddr += 1024;
  }
  WAIT_TENSOR_STORE;
  // flush CB
  dstAddr = (uint64_t)dst + initialCacheLine*64;
  while (minionCacheLines > 0) {
    uint32_t cl = (minionCacheLines >= 16) ? 0xF : minionCacheLines-1;
    // flush L3 to DDR
    evict_va_multi(0x3, dstAddr, cl);
    dstAddr += 1024;
  }
  WAIT_CACHEOPS;
}

GEN_INSTANCES_OP(template, fwdLibSyncopyInstTensorized, void *dst, void *Dims, void *Pitches,
                                  void *src, unsigned int DimNum,
                                  float *scale, int32_t *offset,
                                  unsigned int off);
