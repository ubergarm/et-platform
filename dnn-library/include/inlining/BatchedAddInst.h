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



template <ElemKind dstElK, ElemKind batchElK, ElemKind sliceElK>
inline void fwdLibBatchedAddInstGeneric(LibTensor* outT, LibTensor* in1T,
                                         LibTensor* in2T, uint64_t flags,
                                         const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using dstType = typename elemKind2elemTy<dstElK>::type;
//  using batchType = typename elemKind2elemTy<batchElK>::type;
//  using sliceType = typename elemKind2elemTy<sliceElK>::type;
  
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;


  /* outT --> dst  in1T--> batched  in2T--> slice*/
  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer<void>();
  void* batchT = in1T->getRawDataPointer<void>();
  void* sliceT = in2T->getRawDataPointer<void>();
   
  Addresser<dstElK> tOutput(dstT, outT->getScale(), outT->getOffset());
  const Addresser<batchElK> tBatch(batchT, in1T->getScale(), in1T->getOffset());
  const Addresser<sliceElK> tSlice(sliceT, in2T->getScale(), in2T->getOffset());
  
  // unsigned int *dstIndex = (unsigned int *)pdstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *batchPitch = (unsigned int *)pbatchPitches;
  const dim_t *batchPitch = in1T->strides().data();

  unsigned int pbatchDimNum = static_cast<unsigned int>(in1T->ndims());
  
  unsigned int numElemsDst =
      dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);

  if (maxRead == 0)
    return;

  unsigned int coord[pbatchDimNum];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, pbatchDimNum, dstPitch,
                           dstIndex, k);


  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += batchPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  Operator<Addresser<batchElK>, Addresser<sliceElK>, Addresser<dstElK>, Add> op;

  while (!done && (offsetOut < posMax)) {
    uint64_t offsetIn2 = offsetIn - coord[0]*batchPitch[0];
    op.doOp(tOutput, tBatch, tSlice, offsetOut, offsetIn, offsetIn2);
    done = getOffsets(pbatchDimNum, coord, offsetOut, offsetIn, dstIndex, dstPitch, batchPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}



inline void fwdLibBatchedAddInsti8i32(LibTensor* outT, LibTensor* in1T,
                                              LibTensor* in2T, uint64_t flags,
                                              const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;


  /* outT --> dst  in1T--> batched  in2T--> slice*/
  /* maintain compatibility through the new Iface Libtensor */
  void *dstT = outT->getRawDataPointer<void>();
  
  // int8_t *tOutput = (int8_t *)pdst;
  int8_t *tOutput = outT->getRawDataPointer<int8_t>();
  //  int8_t *tBatch = (int8_t *)pbatch;   // scale[0],offset[0]);
  int8_t *tBatch = in1T->getRawDataPointer<int8_t>();
  //  int32_t *tSlice = (int32_t *)pslice; // scale[1]
  int32_t *tSlice = in2T->getRawDataPointer<int32_t>();
  //  unsigned int *dstIndex = (unsigned int *)pdstDims;
  const dim_t *dstIndex = outT->dims().data();
  //  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch = outT->strides().data();
  //  unsigned int *batchPitch = (unsigned int *)pbatchPitches;
  const dim_t *batchPitch = in1T->strides().data();
  //  unsigned int *slicePitch = (unsigned int *)pslicePitches;
  const dim_t *slicePitch = in2T->strides().data();

  unsigned int pbatchDimNum = static_cast<unsigned int>(in1T->ndims());

  
  float invDstScale;
  getReciprocal(outT->getScale(), invDstScale);

  float invLargeScale = (1 << 15);
  float largeScale;
  getReciprocal(invLargeScale, largeScale);

  unsigned int numElemsDst =
      dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  getCachelinePartition(sizeof(int8_t), numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  unsigned int coord[pbatchDimNum];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, pbatchDimNum, dstPitch,
                           dstIndex, k);

  uint64_t offsetIn2 = 0;
  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;

  unsigned int eSlicePitch[pbatchDimNum];
  eSlicePitch[0] = 0;
  for(unsigned int i = 1; i < pbatchDimNum; i++) eSlicePitch[i] = slicePitch[i - 1];

  for (unsigned int j = 0; j < k; j++) {
    offsetIn2 += eSlicePitch[j] * coord[j];
    offsetIn += batchPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;


  while (!done && (offsetOut < posMax)) {
    int32_t batchVal = tBatch[offsetIn];
    int32_t sliceVal = tSlice[offsetIn2];

    int32_t B = nearbyintf(float(batchVal - in1T->getOffset()) *
                           (in1T->getScale() * invLargeScale));
    int32_t S = nearbyintf(float(sliceVal - in2T->getOffset()) *
                           (in2T->getScale() * invLargeScale));
    int32_t R = B + S;
    tOutput[offsetOut] = clip<int32_t, int8_t>(nearbyintf(float(R) *
                                                          (largeScale * invDstScale) + outT->getOffset()));

    done = getOffsets(pbatchDimNum, coord, offsetOut, offsetIn, offsetIn2, dstIndex, dstPitch, batchPitch, eSlicePitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = (maxRead * sizeof(int8_t) + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + sizeof(int8_t)*initialAddr, clperminion);
}


template <ElemKind dstElK, ElemKind batchElK, ElemKind sliceElK>
inline void fwdLibBatchedAddInst(LibTensor* outT, LibTensor* in1T,
                                   LibTensor* in2T,
                                   uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  if ( dstElK == Int8QTy && batchElK == Int8QTy && sliceElK == Int32QTy)
    inlining::fwdLibBatchedAddInsti8i32(outT, in1T, in2T, flags, minionOffset, assignedMinions);
  else
    inlining::fwdLibBatchedAddInstGeneric<dstElK, batchElK, sliceElK>(outT, in1T, in2T, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _BATCHED_ADD_INST_H_
