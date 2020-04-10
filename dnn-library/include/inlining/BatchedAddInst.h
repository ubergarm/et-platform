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

namespace dnn_lib {

namespace inlining {

template <typename srcType>
inline void fwdLibBatchedAddInst(void *pdst, void *pdstDims,
                                   void *pdstPitches, void *pbatch,
                                   void *pbatchDims, void *pbatchPitches,
                                   unsigned int pbatchDimNum, void *pslice,
                                   const float *scale, const int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tBatch(pbatch, scale[0], offset[0]);
  const Addresser<srcType> tSlice(pslice, scale[1], offset[1]);

  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eBatchPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < pbatchDimNum; i++) {
    eBatchDims[i] = batchIndex[i];
    eDstPitch[i] = dstPitch[i];
    eBatchPitch[i] = batchPitch[i];
  }

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Add> op;
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
              uint64_t srcAddr2 = y * eBatchPitch[1] + z * eBatchPitch[2] +
                                  w * eBatchPitch[3] + q * eBatchPitch[4] +
                                  r * eBatchPitch[5];
              op.doOp(tOutput, tBatch, tSlice, dstAddr, srcAddr1, srcAddr2);
            }
          }
        }
      }
    }
  }
}


template <typename srcType>
inline void fwdLibBatchedAddInstThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, void *pbatch,
    void *pbatchDims, void *pbatchPitches, unsigned int pbatchDimNum,
    void *pslice, const float *scale, const int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tBatch(pbatch, scale[0], offset[0]);
  const Addresser<srcType> tSlice(pslice, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)pdstDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  unsigned int numElemsDst =
      dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
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

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Add> op;

  while (!done && (offsetOut < posMax)) {
    uint64_t offsetIn2 = offsetIn - coord[0]*batchPitch[0];
    op.doOp(tOutput, tBatch, tSlice, offsetOut, offsetIn, offsetIn2);
    done = getOffsets(pbatchDimNum, coord, offsetOut, offsetIn, dstIndex, dstPitch, batchPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);
}

// TODO: Special implementation to support int8_t + int32_t sum, as a quick fix
// we implement this function to support it, the correct way is extend the
// BatchedAdd templatized op and the Operator class in order to support 2
// different templates
inline void fwdLibBatchedAddInsti8i32(void *pdst, void *pdstDims,
                                        void *pdstPitches, void *pbatch,
                                        void *pbatchDims, void *pbatchPitches,
                                        unsigned int pbatchDimNum, void *pslice,
                                        void *pslicePitches, const float *scale,
                                        const int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tBatch = (int8_t *)pbatch;   // scale[0],offset[0]);
  int32_t *tSlice = (int32_t *)pslice; // scale[1]

  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;
  unsigned int *slicePitch = (unsigned int *)pslicePitches;

  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);

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
  getReciprocal(scale[2], invDstScale);

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

              int32_t B = nearbyintf(float(batchVal - offset[0]) *
                                     (scale[0] * invLargeScale));
              int32_t S = nearbyintf(float(sliceVal - offset[1]) *
                                     (scale[1] * invLargeScale));
              int32_t R = B + S;
              tOutput[dstAddr] = clip<int32_t, int8_t>(nearbyintf(
                  float(R) * (largeScale * invDstScale) + offset[2]));
            }
          }
        }
      }
    }
  }
}


inline void fwdLibBatchedAddInsti8i32Threaded(void *pdst, void *pdstDims,
                                                void *pdstPitches, void *pbatch,
                                                void *pbatchDims, void *pbatchPitches,
                                                unsigned int pbatchDimNum, void *pslice,
                                                void *pslicePitches, const float *scale,
                                                const int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

 int8_t *tOutput = (int8_t *)pdst;
  int8_t *tBatch = (int8_t *)pbatch;   // scale[0],offset[0]);
  int32_t *tSlice = (int32_t *)pslice; // scale[1]

  unsigned int *dstIndex = (unsigned int *)pdstDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;
  unsigned int *slicePitch = (unsigned int *)pslicePitches;

  float invDstScale;
  getReciprocal(scale[2], invDstScale);

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

    int32_t B = nearbyintf(float(batchVal - offset[0]) *
                                     (scale[0] * invLargeScale));
    int32_t S = nearbyintf(float(sliceVal - offset[1]) *
                                     (scale[1] * invLargeScale));
    int32_t R = B + S;
    tOutput[offsetOut] = clip<int32_t, int8_t>(nearbyintf(
    float(R) * (largeScale * invDstScale) + offset[2]));

    done = getOffsets(pbatchDimNum, coord, offsetOut, offsetIn, offsetIn2, dstIndex, dstPitch, batchPitch, eSlicePitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * sizeof(int8_t) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + sizeof(int8_t)*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _BATCHED_ADD_INST_H_
