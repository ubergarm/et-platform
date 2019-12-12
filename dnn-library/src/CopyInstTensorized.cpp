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

 // This implementation takes advantage of small cases with the same input and
 // output shape. It does not try to avoid padding, as the calculations needed
 // would decrease velocity. Therefore, we just give each minion its initial
 // address and make it copy everything until maxRead. In the end, there should
 // be a graph decision between this version and the general vectorisation.

 template <typename srcType>
 void dnn_lib::fwdLibCopyInstTensorized(void *dst, void *dstDims, void *dstPitches,
                                        void *src, void *srcDims, void *srcPitches,
                                        unsigned int srcDimNum, float *scale,
                                        int32_t *offset, uint64_t flags,
                                        const uint32_t minionOffset,
                                        const uint32_t assignedMinions) {

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (32 * ACTIVE_SHIRES) : assignedMinions;
  if ((minionId >= activeMinions) || (minionId >= activeMinions))
    return;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  size_t typeSize = getsize<srcType>();
  uint64_t numElemsDst = dstPitch[0] * actIndex[0] *
                             typeSize; // Total number of elements in the tensor
  uint64_t numCacheLines = (numElemsDst - 1) / 64 + 1; //64 = CacheLineLength
  uint64_t minionCacheLines = (numCacheLines - 1) / activeMinions + 1;
  uint64_t initialCacheLine = minionCacheLines * minionId;
  uint64_t lastCacheLine = initialCacheLine + minionCacheLines;
  minionCacheLines =
          (lastCacheLine <= numCacheLines) ? minionCacheLines
        : (initialCacheLine < numCacheLines) ? numCacheLines - initialCacheLine : 0;
  uint64_t srcAddr = (uint64_t)src + initialCacheLine*64;
  uint64_t dstAddr = (uint64_t)dst + initialCacheLine*64;

  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

  while (minionCacheLines >= 16) {
    tensor_load(0, 0, 0, 0, 0, srcAddr, 0, 0xF, 0x40, 0);
    tensor_store_scp(0, 0, 0xF, dstAddr, 0x40);
    srcAddr += 1024;
    dstAddr += 1024;
    minionCacheLines -= 16;
  }
  if (minionCacheLines == 0) return;
  tensor_load(0, 0, 0, 0, 0, srcAddr, 0, minionCacheLines-1, 0x40, 0);
  tensor_store_scp(0, 0, minionCacheLines-1, dstAddr, 0x40);
}

GEN_INSTANCES_OP(template, fwdLibCopyInstTensorized, void *dst, void *dstDims, void *dstPitches,
                                  void *src, void *srcDims, void *srcPitches,
                                  unsigned int srcDimNum,
                                  float *scale, int32_t *offset, uint64_t flags,
                                  const uint32_t minionOffset, const uint32_t assignedMinions);
