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
                                 LibTensor* in2T,
                                 uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
//  using dstType = typename elemKind2elemTy<dstElK>::type;
//  using batchType = typename elemKind2elemTy<batchElK>::type;
//  using sliceType = typename elemKind2elemTy<sliceElK>::type;
  
  if (get_minion_id() != minionOffset) return;

  /* outT --> dst  in1T--> batched  in2T--> slice*/
  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer<void>();
  void* batchT = in1T->getRawDataPointer<void>();
  void* sliceT = in2T->getRawDataPointer<void>();
  
  Addresser<dstElK> tOutput(dstT, outT->getScale(), outT->getOffset());
  const Addresser<batchElK> tBatch(batchT, in1T->getScale(), in1T->getOffset());
  const Addresser<sliceElK> tSlice(sliceT, in2T->getScale(), in2T->getOffset());

  assert(in1T->ndims() <= MAX_TENSOR_DIMENSIONS);

  const dim_array_t & eBatchDims = in1T->dims();
  const dim_array_t & eDstPitch = outT->strides();
  const dim_array_t & eBatchPitch = in1T->strides();
  const dim_array_t & eSlicePitch = in2T->strides();


  Operator<Addresser<batchElK>, Addresser<sliceElK>, Addresser<dstElK>, Add> op;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              // tOutput[x,y,z,w,q,r] = tBatch[x,y,z,w,q,r] + tSlice[y,z,w,q,r];
              uint64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                 z * eDstPitch[2] + w * eDstPitch[3] +
                                 q * eDstPitch[4] + r * eDstPitch[5];
              uint64_t srcAddr1 = x * eBatchPitch[0] + y * eBatchPitch[1] +
                                  z * eBatchPitch[2] + w * eBatchPitch[3] +
                                  q * eBatchPitch[4] + r * eBatchPitch[5];
              uint64_t srcAddr2 = y * eSlicePitch[0] + z * eSlicePitch[1] +
                                  w * eSlicePitch[2] + q * eSlicePitch[3] +
                                  r * eSlicePitch[4];              
              op.doOp(tOutput, tBatch, tSlice, dstAddr, srcAddr1, srcAddr2);
            }
          }
        }
      }
    }
  }
}


template <ElemKind dstElK, ElemKind batchElK, ElemKind sliceElK>
inline void fwdLibBatchedAddInstThreadedGeneric(LibTensor* outT, LibTensor* in1T,
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
  getUniformCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                               activeMinions);
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
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}



// TODO: Special implementation to support int8_t + int32_t sum, as a quick fix
// we implement this function to support it, the correct way is extend the
// BatchedAdd templatized op and the Operator class in order to support 2
// different templates
inline void fwdLibBatchedAddInsti8i32(LibTensor* outT, LibTensor* in1T,
                                      LibTensor* in2T,
                                      uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;
  
  /* outT --> dst  in1T--> batched  in2T--> slice*/
  /* maintain compatibility through the new Iface Libtensor */
  // int8_t *tOutput = (int8_t *)pdst;
  int8_t *tOutput = outT->getRawDataPointer<int8_t>();
  // int8_t *tBatch = (int8_t *)pbatch;   // scale[0],offset[0]);
  int8_t *tBatch = in1T->getRawDataPointer<int8_t>();
  // int32_t *tSlice = (int32_t *)pslice; // scale[1]
  int32_t *tSlice = in2T->getRawDataPointer<int32_t>();
  // unsigned int *batchIndex = (unsigned int *)pbatchDims;
  const dim_t *batchIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *batchPitch = (unsigned int *)pbatchPitches;
  const dim_t *batchPitch = in1T->strides().data();
  // unsigned int *slicePitch = (unsigned int *)pslicePitches;
  const dim_t *slicePitch = in2T->strides().data();

  unsigned int pbatchDimNum = static_cast<unsigned int>(in1T->ndims());
  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);
  assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eBatchPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSlicePitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < pbatchDimNum; i++) {
    eBatchDims[i] = batchIndex[i];
    eDstPitch[i] = dstPitch[i];
    eBatchPitch[i] = batchPitch[i];
    eSlicePitch[i] = slicePitch[i];
  }
  float invDstScale;
  getReciprocal(outT->getScale(), invDstScale);

  float invLargeScale = (1 << 15);
  float largeScale;
  getReciprocal(invLargeScale, largeScale);
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              // tOutput[x,y,z,w,q,r] = tBatch[x,y,z,w,q,r] + tSlice[y,z,w,q,r];
              uint64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                 z * eDstPitch[2] + w * eDstPitch[3] +
                                 q * eDstPitch[4] + r * eDstPitch[5];
              uint64_t srcAddr1 = x * eBatchPitch[0] + y * eBatchPitch[1] +
                                  z * eBatchPitch[2] + w * eBatchPitch[3] +
                                  q * eBatchPitch[4] + r * eBatchPitch[5];
              uint64_t srcAddr2 = y * eSlicePitch[0] + z * eSlicePitch[1] +
                                  w * eSlicePitch[2] + q * eSlicePitch[3] +
                                  r * eSlicePitch[4];

              int32_t batchVal = tBatch[srcAddr1];
              int32_t sliceVal = tSlice[srcAddr2];
              int32_t B = nearbyintf(float(batchVal - in1T->getOffset()) *
                                     (in1T->getScale() * invLargeScale));
              int32_t S = nearbyintf(float(sliceVal - in2T->getOffset()) *
                                     (in2T->getScale() * invLargeScale));
              int32_t R = B + S;
              tOutput[dstAddr] = clip<int32_t, int8_t>(nearbyintf(float(R) *
                                                                  (largeScale * invDstScale) + outT->getOffset()));
            }
          }
        }
      }
    }
  }
}


inline void fwdLibBatchedAddInsti8i32Threaded(LibTensor* outT, LibTensor* in1T,
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
  getUniformCachelinePartition(sizeof(int8_t), numElemsDst, initialAddr, maxRead,
                               activeMinions);
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
  unsigned int clperminion = maxRead * sizeof(int8_t) / CACHE_LINE_BYTES;
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

  template <ElemKind dstElK, ElemKind batchElK, ElemKind sliceElK>
inline void fwdLibBatchedAddInstThreaded(LibTensor* outT, LibTensor* in1T,
                                   LibTensor* in2T,
                                   uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  if ( dstElK == Int8QTy && batchElK == Int8QTy && sliceElK == Int32QTy)
    inlining::fwdLibBatchedAddInsti8i32Threaded(outT, in1T, in2T, flags, minionOffset, assignedMinions);
  else
    inlining::fwdLibBatchedAddInstThreadedGeneric<dstElK, batchElK, sliceElK>(outT, in1T, in2T, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _BATCHED_ADD_INST_H_
