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

template <typename srcType>
void dnn_lib::fwdLibBatchedReduceAddInst(void *pdst, void *pdstDims,
                                         void *pdstPitches, void *pbatch,
                                         void *pbatchDims, void *pbatchPitches,
                                         unsigned int pbatchDimNum,
                                         unsigned int axis, float *scale,
                                         int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(pdst, scale[1], offset[1]);
  const Addresser<srcType> tBatch(pbatch, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eBatchPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < pbatchDimNum; i++) {
    eBatchDims[i] = batchIndex[i];
    if (i < axis) {
      eDstPitch[i] = dstPitch[i];
    } else if (i > axis) {
      eDstPitch[i] = dstPitch[i - 1];
    }
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
              uint64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                 z * eDstPitch[2] + w * eDstPitch[3] +
                                 q * eDstPitch[4] + r * eDstPitch[5];
              uint64_t srcAddr = x * eBatchPitch[0] + y * eBatchPitch[1] +
                                 z * eBatchPitch[2] + w * eBatchPitch[3] +
                                 q * eBatchPitch[4] + r * eBatchPitch[5];
              op.doOp(tOutput, tOutput, tBatch, dstAddr, dstAddr, srcAddr);
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibBatchedReduceAddInstThreaded(void *pdst, void *pdstDims,
                                                 void *pdstPitches, void *pbatch,
                                                 void *pbatchDims, void *pbatchPitches,
                                                 unsigned int pbatchDimNum,
                                                 unsigned int axis, float *scale,
                                                 int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(pdst, scale[1], offset[1]);
  const Addresser<srcType> tBatch(pbatch, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  unsigned int numElemsDst;

  numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);

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
  Addresser<srcType> *tOutputPtr;
  Addresser<srcType> *tSumPtr;
  bool done = false;
  //int sum = 0;
  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Add> op;
  while (!done && (offsetOut < posMax)) {
    tOutput[offsetOut] = tBatch[offsetIn];
    offsetIn += batchPitch[axis];
    for (size_t i = 1; i < batchIndex[axis]; i++) {
      print(__PRETTY_FUNCTION__);
      Addresser<srcType> tSum = tOutput;
      op.doOp(tOutput, tSum, tBatch, offsetOut, offsetOut, offsetIn);
      offsetIn += batchPitch[axis];
    }
    offsetIn -= batchIndex[axis] * batchPitch[axis];

    done = getOffsets(pbatchDimNum - 1, offsets, offsetIn, offsetOut, dstIndex,
                      redBatchPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);
}

void dnn_lib::fwdLibBatchedReduceAddInstInt8(
    void *pdst, void *pdstDims, void *pdstPitches, void *pbatch,
    void *pbatchDims, void *pbatchPitches, unsigned int pbatchDimNum,
    unsigned int axis, float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tBatch = (int8_t *)pbatch;

  float invScale;
  getReciprocal(scale[1], invScale);
  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eBatchPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < pbatchDimNum; i++) {
    eBatchDims[i] = batchIndex[i];
    if (i < axis) {
      eDstPitch[i] = dstPitch[i];
    } else if (i > axis) {
      eDstPitch[i] = dstPitch[i - 1];
    }
    eBatchPitch[i] = batchPitch[i];
  }
#define LOOP_AXIS_CASE(_D0, _D1, _D2, _D3, _D4, _D5_AXIS)                      \
  for (size_t i##_D0 = 0; i##_D0 < eBatchDims[_D0]; i##_D0++)                  \
    for (size_t i##_D1 = 0; i##_D1 < eBatchDims[_D1]; i##_D1++)                \
      for (size_t i##_D2 = 0; i##_D2 < eBatchDims[_D2]; i##_D2++)              \
        for (size_t i##_D3 = 0; i##_D3 < eBatchDims[_D3]; i##_D3++)            \
          for (size_t i##_D4 = 0; i##_D4 < eBatchDims[_D4]; i##_D4++) {        \
            float sum = 0.0;                                                   \
            for (size_t i##_D5_AXIS = 0; i##_D5_AXIS < eBatchDims[_D5_AXIS];   \
                 i##_D5_AXIS++) {                                              \
              uint64_t srcAddr = i0 * eBatchPitch[0] + i1 * eBatchPitch[1] +   \
                                 i2 * eBatchPitch[2] + i3 * eBatchPitch[3] +   \
                                 i4 * eBatchPitch[4] + i5 * eBatchPitch[5];    \
              sum += tBatch[srcAddr] - offset[0];                              \
            }                                                                  \
            size_t i##_D5_AXIS = 0;                                            \
            int32_t res = nearbyintf(sum * scale[0] * invScale) + offset[1];   \
            uint64_t dstAddr = i0 * eDstPitch[0] + i1 * eDstPitch[1] +         \
                               i2 * eDstPitch[2] + i3 * eDstPitch[3] +         \
                               i4 * eDstPitch[4] + i5 * eDstPitch[5];          \
            tOutput[dstAddr] = clip<int32_t, int8_t>(res);                     \
          }
  // Each loop order, with the inner-most dimension/index equal to the axis.
  switch (axis) {
  case 0:
    LOOP_AXIS_CASE(1, 2, 3, 4, 5, 0);
    break;
  case 1:
    LOOP_AXIS_CASE(0, 2, 3, 4, 5, 1);
    break;
  case 2:
    LOOP_AXIS_CASE(0, 1, 3, 4, 5, 2);
    break;
  case 3:
    LOOP_AXIS_CASE(0, 1, 2, 4, 5, 3);
    break;
  case 4:
    LOOP_AXIS_CASE(0, 1, 2, 3, 5, 4);
    break;
  case 5:
    LOOP_AXIS_CASE(0, 1, 2, 3, 4, 5);
    break;
  default: // TODO Add some warning message(axis bigger than num of dims)
    break;
  }
#undef LOOP_AXIS_CASE
}


void dnn_lib::fwdLibBatchedReduceAddInstInt8Threaded(
    void *pdst, void *pdstDims, void *pdstPitches, void *pbatch,
    void *pbatchDims, void *pbatchPitches, unsigned int pbatchDimNum,
    unsigned int axis, float *scale, int32_t *offset, uint64_t flags) {


  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tBatch = (int8_t *)pbatch;

  float invScale;
  getReciprocal(scale[1], invScale);
  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  unsigned int numElemsDst;

  numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  getUniformCachelinePartition(sizeof(int8_t), numElemsDst, initialAddr, maxRead,
                               activeMinions);

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
      // print(__PRETTY_FUNCTION__);
      sum += tBatch[offsetIn] - offset[0];
      offsetIn += batchPitch[axis];
    }
    offsetIn -= batchIndex[axis] * batchPitch[axis];
    int32_t res = nearbyintf(sum * scale[0] * invScale) + offset[1];
    tOutput[offsetOut] = clip<int32_t, int8_t>(res);

    done = getOffsets(pbatchDimNum - 1, offsets, offsetIn, offsetOut, dstIndex,
                      redBatchPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * sizeof(int8_t) / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + sizeof(int8_t)*initialAddr, clperminion);
}

GEN_INSTANCES_OP(template, fwdLibBatchedReduceAddInst, void *pdst, void *pdstDims, void *pdstPitches,
                                   void *pbatch, void *pbatchDims, void *pbatchPitches,
                                   unsigned int pbatchDimNum, unsigned int axis,
                                   float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibBatchedReduceAddInstThreaded, void *pdst, void *pdstDims, void *pdstPitches,
                                   void *pbatch, void *pbatchDims, void *pbatchPitches,
                                   unsigned int pbatchDimNum, unsigned int axis,
                                   float *scale, int32_t *offset, uint64_t flags);
