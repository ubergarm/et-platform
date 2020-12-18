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

#ifndef _BATCHED_REDUCE_ADD_INST_H_
#define _BATCHED_REDUCE_ADD_INST_H_

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
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {


template <ElemKind elK>
inline void fwdLibBatchedReduceAddInst(LibTensor* outT, LibTensor* inT,
                                       unsigned int axis,  uint64_t flags,
                                       const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer<void>();
  void* batchT = inT->getRawDataPointer<void>();

  // Addresser<elK> tOutput(pdst, scale[1], offset[1]);
  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tBatch(pbatch, scale[0], offset[0]);
  const Addresser<elK> tBatch(batchT, inT->getScale(), inT->getOffset());

  // unsigned int *dstIndex = (unsigned int *)pdstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *batchIndex = (unsigned int *)pbatchDims;
  const dim_t *batchIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *batchPitch = (unsigned int *)pbatchPitches;
  const dim_t *batchPitch = inT->strides().data();

  unsigned int pbatchDimNum = static_cast<unsigned int>(inT->ndims());
  unsigned int numElemsDst;

  numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);

  if (maxRead == 0)
    return;

  unsigned int offsets[pbatchDimNum - 1];

  unsigned int k;

  unsigned int redBatchPitch[pbatchDimNum - 1];
  for (size_t i = 0; i < pbatchDimNum; i++) {
    if (i < axis) {
      redBatchPitch[i] = batchPitch[i];

    } else if (i > axis) {
      redBatchPitch[i - 1] = batchPitch[i];
    }
  }

  getNonPaddingCoordinates(offsets, initialAddr, pbatchDimNum - 1, dstPitch, dstIndex,
                           k);
  uint64_t offsetOut = 0;
  uint64_t offsetIn = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * offsets[j];
    offsetIn += redBatchPitch[j] * offsets[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  //int sum = 0;
  Operator<Addresser<elK>, Addresser<elK>, Addresser<elK>, Add> op;
  while (!done && (offsetOut < posMax)) {
    tOutput[offsetOut] = tBatch[offsetIn];
    offsetIn += batchPitch[axis];
    for (size_t i = 1; i < batchIndex[axis]; i++) {
      Addresser<elK> tSum = tOutput;
      op.doOp(tOutput, tSum, tBatch, offsetOut, offsetOut, offsetIn);
      offsetIn += batchPitch[axis];
    }
    offsetIn -= batchIndex[axis] * batchPitch[axis];

    done = getOffsets(pbatchDimNum - 1, offsets, offsetIn, offsetOut, dstIndex,
                      redBatchPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}


template<>
inline void fwdLibBatchedReduceAddInst<Int8QTy>(LibTensor* outT,
                                                LibTensor* inT,
                                                unsigned int axis,
                                                uint64_t flags, const uint32_t minionOffset,
                                                const uint32_t assignedMinions) {
    
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  
  void *dstT = outT->getRawDataPointer<void>();
  
  // int8_t *tOutput = (int8_t *)pdst;
  int8_t *tOutput = outT->getRawDataPointer<int8_t>();
  // int8_t *tBatch = (int8_t *)pbatch;
  int8_t *tBatch = inT->getRawDataPointer<int8_t>();
  
  float invScale;
  getReciprocal(outT->getScale(), invScale);
  // unsigned int *dstIndex = (unsigned int *)pdstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *batchIndex = (unsigned int *)pbatchDims;
  const dim_t *batchIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *batchPitch = (unsigned int *)pbatchPitches;
  const dim_t *batchPitch = inT->strides().data();

  unsigned int pbatchDimNum = static_cast<unsigned int>(inT->ndims());

  unsigned int numElemsDst;

  numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  getCachelinePartition(sizeof(int8_t), numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);

  if (maxRead == 0)
    return;

  unsigned int offsets[pbatchDimNum - 1];

  unsigned int k;

  unsigned int redBatchPitch[pbatchDimNum - 1];
  for (size_t i = 0; i < pbatchDimNum; i++) {
    if (i < axis) {
      redBatchPitch[i] = batchPitch[i];

    } else if (i > axis) {
      redBatchPitch[i - 1] = batchPitch[i];
    }
  }

  getNonPaddingCoordinates(offsets, initialAddr, pbatchDimNum - 1, dstPitch, dstIndex,
                           k);
  uint64_t offsetOut = 0;
  uint64_t offsetIn = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * offsets[j];
    offsetIn += redBatchPitch[j] * offsets[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (not done && offsetOut < posMax) {
    float sum = 0.0;
    for (size_t i = 0; i < batchIndex[axis]; i++) {
      sum += tBatch[offsetIn] - inT->getOffset();
      offsetIn += batchPitch[axis];
    }
    offsetIn -= batchIndex[axis] * batchPitch[axis];
    int32_t res = nearbyintf(sum * inT->getScale() * invScale) + outT->getOffset();
    tOutput[offsetOut] = clip<int32_t, int8_t>(res);

    done = getOffsets(pbatchDimNum - 1, offsets, offsetIn, offsetOut, dstIndex,
                      redBatchPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = (maxRead * sizeof(int8_t) + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + sizeof(int8_t)*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _BATCHED_REDUCE_ADD_INST_H_
