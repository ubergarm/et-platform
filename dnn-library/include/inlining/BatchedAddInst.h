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

#ifndef _BATCHED_ADD_INST_H_
#define _BATCHED_ADD_INST_H_

#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

#include "Addresser.h" // From include/internal path
#include "Float16.h"
#include "LibTensor.h"
#include "Operator.h"
#include "Writer.h" // From include/internal path
#include "utils.h"  // From include/internal path

namespace dnn_lib {

namespace inlining {

template <ElemKind dstElK, ElemKind batchElK, ElemKind sliceElK>
INLINE_ATTR void fwdLibBatchedAddInstGeneric(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,
                                             const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using dstType = typename elemKind2elemTy<dstElK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* outT --> dst  in1T--> batched  in2T--> slice*/
  void* dstT = outT->getRawDataPointer();
  void* batchT = in1T->getRawDataPointer();
  void* sliceT = in2T->getRawDataPointer();

  Addresser<dstElK> tOutput(dstT, outT->getScale(), outT->getOffset());
  const Addresser<batchElK> tBatch(batchT, in1T->getScale(), in1T->getOffset());
  const Addresser<sliceElK> tSlice(sliceT, in2T->getScale(), in2T->getOffset());
  
  const dim_t *dstIndex = outT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *batchPitch = in1T->strides().data();
  const dim_t* slicePitch = in2T->strides().data();

  const dim_t pbatchDimNum = in1T->ndims();

  size_t numElemsDst = dstPitch[0] * dstIndex[0];

  dim_t eSlicePitch[pbatchDimNum];
  eSlicePitch[0] = 0;
  for (dim_t i = 1; i < pbatchDimNum; i++) {
    eSlicePitch[i] = slicePitch[i - 1];
  }

  size_t initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);

  if (maxRead == 0)
    return;

  dim_array_t coord = {0};
  dim_t k = 0;
  getNonPaddingCoordinates(coord, initialAddr, pbatchDimNum, dstPitch,
                           dstIndex, k);

  size_t offsetIn = 0;
  size_t offsetIn2 = 0;
  size_t offsetOut = 0;
  for (dim_t j = 0; j < k; j++) {
    offsetIn += batchPitch[j] * coord[j];
    offsetIn2 += eSlicePitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  auto posMax = maxRead + initialAddr;
  bool done = false;

  Operator<Addresser<batchElK>, Addresser<sliceElK>, Addresser<dstElK>, Add> op;

  while (!done && (offsetOut < posMax)) {
    op.doOp(tOutput, tBatch, tSlice, offsetOut, offsetIn, offsetIn2);
    done = getOffsets(pbatchDimNum, coord, offsetOut, offsetIn, offsetIn2, dstIndex, dstPitch, batchPitch, eSlicePitch);
  }

  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

INLINE_ATTR void fwdLibBatchedAddInsti8i32(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,
                                           const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;


  /* outT --> dst  in1T--> batched  in2T--> slice*/
  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer();

  auto tOutput = outT->getRawDataPointer<int8_t>();
  auto tBatch = in1T->getRawDataPointer<int8_t>();
  auto tSlice = in2T->getRawDataPointer<int32_t>();
  const dim_t *dstIndex = outT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *batchPitch = in1T->strides().data();
  const dim_t *slicePitch = in2T->strides().data();

  dim_t pbatchDimNum = in1T->ndims();

  float invDstScale;
  getReciprocal(outT->getScale(), invDstScale);

  float invLargeScale = (1 << 15);
  float largeScale;
  getReciprocal(invLargeScale, largeScale);

  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  getCachelinePartition(sizeof(int8_t), numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  dim_array_t coord = {0};
  dim_t k = 0;
  getNonPaddingCoordinates(coord, initialAddr, pbatchDimNum, dstPitch,
                           dstIndex, k);

  size_t offsetIn2 = 0;
  size_t offsetIn = 0;
  size_t offsetOut = 0;

  dim_t eSlicePitch[pbatchDimNum];
  eSlicePitch[0] = 0;
  for (dim_t i = 1; i < pbatchDimNum; i++) {
    eSlicePitch[i] = slicePitch[i - 1];
  }

  for (dim_t j = 0; j < k; j++) {
    offsetIn2 += eSlicePitch[j] * coord[j];
    offsetIn += batchPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  auto posMax = maxRead + initialAddr;
  bool done = false;


  while (!done && (offsetOut < posMax)) {
    int32_t batchVal = tBatch[offsetIn];
    int32_t sliceVal = tSlice[offsetIn2];

    int32_t B =
      static_cast<int32_t>(nearbyintf(float(batchVal - in1T->getOffset()) * (in1T->getScale() * invLargeScale)));
    int32_t S =
      static_cast<int32_t>(nearbyintf(float(sliceVal - in2T->getOffset()) * (in2T->getScale() * invLargeScale)));
    int32_t R = B + S;
    tOutput[offsetOut] = clip<int32_t, int8_t>(
      static_cast<int32_t>(nearbyintf(float(R) * (largeScale * invDstScale) + static_cast<float>(outT->getOffset()))));

    done = getOffsets(pbatchDimNum, coord, offsetOut, offsetIn, offsetIn2, dstIndex, dstPitch, batchPitch, eSlicePitch);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * sizeof(int8_t) + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + sizeof(int8_t)*initialAddr, clperminion);
}

template <ElemKind dstElK, ElemKind batchElK, ElemKind sliceElK>
INLINE_ATTR void fwdLibBatchedAddInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,
                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  if ( dstElK == Int8QTy && batchElK == Int8QTy && sliceElK == Int32QTy)
    inlining::fwdLibBatchedAddInsti8i32(outT, in1T, in2T, flags, minionOffset, assignedMinions);
  else
    inlining::fwdLibBatchedAddInstGeneric<dstElK, batchElK, sliceElK>(outT, in1T, in2T, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _BATCHED_ADD_INST_H_
