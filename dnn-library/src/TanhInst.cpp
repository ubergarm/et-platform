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

// TODO Check corner cases
template <typename srcType>
void dnn_lib::fwdLibTanhInst(void *dstT, void *dstDims, void *dstPitches,
                             void *srcT1, void *srcDims, void *srcPitches,
                             unsigned int srcDimNum, float *scale,
                             int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> ptrSrcT1(srcT1, scale[0], offset[0]);
  Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  float op1, op2;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              op1 = getSinh(ptrSrcT1[x * eSrcPitch[0] + y * eSrcPitch[1] +
                                     z * eSrcPitch[2] + w * eSrcPitch[3] +
                                     q * eSrcPitch[4] + r * eSrcPitch[5]]);
              op2 = getCosh(ptrSrcT1[x * eSrcPitch[0] + y * eSrcPitch[1] +
                                     z * eSrcPitch[2] + w * eSrcPitch[3] +
                                     q * eSrcPitch[4] + r * eSrcPitch[5]]);
              fpReciprocalSingleElement(op2, op2);
              ptrDstT[x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                      w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5]] =
                  op1 * op2;
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibTanhInstThreaded(void *dstT, void *dstDims,
                                     void *dstPitches, void *srcT1,
                                     void *srcDims, void *srcPitches,
                                     unsigned int srcDimNum,
                                     float *scale, int32_t *offset,
                                     uint64_t flags) {


  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  float op1, op2;
  while (!done && (offsetOut < posMax)) {
    op1 = getSinh(aSrcT1[offsetIn]);
    op2 = getCosh(aSrcT1[offsetIn]);
    fpReciprocalSingleElement(op2, op2);
    ptrDstT[offsetOut] = op1 * op2;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

GEN_INSTANCES_OP(template, fwdLibTanhInst, void *dstT, void *dstDims, void *dstPitches, void *srcT1,
                       void *srcDims, void *srcPitches, unsigned int srcDimNum,
                       float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibTanhInstThreaded, void *dstT, void *dstDims, void *dstPitches, void *srcT1,
                       void *srcDims, void *srcPitches, unsigned int srcDimNum,
                       float *scale, int32_t *offset, uint64_t flags);
