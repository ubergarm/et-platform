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
void dnn_lib::fwdLibModuloInst(void *dstT, void *dstDims, void *dstPitches,
                               void *srcT, void *srcDims, void *srcPitches,
                               unsigned int srcDimNum, long long divisor,
                               bool signFollowDivisor, float *scale,
                               int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  srcType *tOutput = (srcType *)dstT;
  srcType *tInput = (srcType *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;

  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              auto res = (tInput[addrSrc]) % divisor;
              if (signFollowDivisor && (res < 0)) {
                res += divisor;
              }
              tOutput[addrDst] = res;
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibModuloInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, long long divisor,
    bool signFollowDivisor, float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);

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

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    auto res = (tInput[offsetIn]) % divisor;
    if (signFollowDivisor && (res < 0)) {
      res += divisor;
    }
    tOutput[offsetOut] = res;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

GEN_INSTANCES_INTONLY_OP(template, fwdLibModuloInst, void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, long long divisor, bool signFollowDivisor,
                                 float * scale, int32_t * offset);
GEN_INSTANCES_INTONLY_OP(template, fwdLibModuloInstThreaded, void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, long long divisor, bool signFollowDivisor,
                                 float * scale, int32_t * offset, uint64_t flags);
