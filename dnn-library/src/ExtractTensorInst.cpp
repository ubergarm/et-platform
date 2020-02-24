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
void dnn_lib::fwdLibExtractTensorInst(void *dst, void *dstDims,
                                      void *dstPitches, unsigned int dstDimNum,
                                      void *src, void *srcDims,
                                      void *srcPitches, void *pcoord,
                                      const float *scale, const int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int *coord = (unsigned int *)pcoord;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eOffsets[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < dstDimNum; i++) {
    eDims[i] = dstIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
    eOffsets[i] = coord[i];
  }

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              tOutput[x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                      w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5]] =
                  tInput[(eOffsets[0] + x) * eSrcPitch[0] +
                         (eOffsets[1] + y) * eSrcPitch[1] +
                         (eOffsets[2] + z) * eSrcPitch[2] +
                         (eOffsets[3] + w) * eSrcPitch[3] +
                         (eOffsets[4] + q) * eSrcPitch[4] +
                         (eOffsets[5] + r) * eSrcPitch[5]];
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibExtractTensorInstThreaded(void *dst, void *dstDims,
                                              void *dstPitches,
                                              unsigned int dstDimNum, void *src,
                                              void *srcDims, void *srcPitches,
                                              void *pcoord, const float *scale,
                                              const int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int *coord = (unsigned int *)pcoord;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialaddrOut, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialaddrOut, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coordOut[dstDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coordOut, initialaddrOut, dstDimNum, dstPitch,
                           dstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++)
    offsetOut += dstPitch[i] * coordOut[i];
  unsigned int offsetIn = 0;
  for (unsigned int i = 0; i < dstDimNum; ++i)
    offsetIn += (coord[i] + coordOut[i]) * srcPitch[i];

  unsigned int posMaxOut = maxRead + initialaddrOut;
  bool done = false;
  while (!done && (offsetOut < posMaxOut)) {
    tOutput[offsetOut] = tInput[offsetIn];
    done = getOffsets(dstDimNum, coordOut, offsetIn, offsetOut, dstIndex,
                      srcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialaddrOut, clperminion);
}

GEN_INSTANCES_OP(template, fwdLibExtractTensorInst, void *dst, void *dstDims,
                                void *dstPitches, unsigned int dstDimNum,
                                void *src2, void *src2Dims, void *src2Pitches,
                                void * poffsets, const float *scale, const int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibExtractTensorInstThreaded, void *dst, void *dstDims,
                                void *dstPitches, unsigned int dstDimNum,
                                void *src, void *srcDims, void *srcPitches,
                                void * poffsets, const float *scale, const int32_t *offset, uint64_t flags);
