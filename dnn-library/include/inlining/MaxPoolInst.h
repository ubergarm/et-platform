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

#ifndef _MAX_POOL_INST_H_
#define _MAX_POOL_INST_H_

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
inline void fwdLibMaxPoolInst(bool argMax, void *dstMatrix, void *dstMatrixDims,
                                void *dstMatrixPitches, void *dst2Matrix,
                                void *dst2MatrixDims, void *dst2MatrixPitches, void *srcMatrixPitchesNoPadding,
                                void *activations, void *activationsDims,
                                void *activationsPitches, void *pkernels,
                                void *pstrides, void *ppads, const float *scale,
                                const int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  int64_t *tOutput2 = (int64_t *)dst2Matrix;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *dst2Pitch = (unsigned int *)dst2MatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *srcPitchNoPadding = (unsigned int *)  srcMatrixPitchesNoPadding;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides;
  unsigned int *pads = (unsigned int *)ppads;

  // For each input in the batch:
  for (size_t n = 0; n < dstIndex[0]; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < actIndex[3]; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pads[0]);
      for (size_t ax = 0; ax < dstIndex[1]; x += strides[0], ax++) {
        ssize_t y = -ssize_t(pads[1]);
        for (size_t ay = 0; ay < dstIndex[2]; y += strides[1], ay++) {

          bool first = true;
          auto max_value = tAInput[0];
          max_value = 0;
          int64_t argmaxNHWC = 0;

          for (size_t fx = 0; fx < kernels[0]; fx++) {
            for (size_t fy = 0; fy < kernels[1]; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
                  oy >= ssize_t(actIndex[2])) {
                continue;
              }

              int64_t idx = n * actPitch[0] + (size_t)ox * actPitch[1] +
                (size_t)oy * actPitch[2] + z;
              auto val = tAInput[idx];
              if (first || (val >= max_value)) {
                first = false;
                max_value = val;
                if (argMax) 
                  argmaxNHWC = n * srcPitchNoPadding[0] +
                    (size_t)ox * srcPitchNoPadding[1] +
                    (size_t)oy * srcPitchNoPadding[2]
                    + z;
              }
            }
          }

          int64_t dstAddr = n * dstPitch[0] + (size_t)ax * dstPitch[1] +
                            (size_t)ay * dstPitch[2] + z;
          tOutput[dstAddr] = max_value;

          if (argMax) {
            int64_t dst2Addr = n * dst2Pitch[0] + (size_t)ax * dst2Pitch[1] +
                               (size_t)ay * dst2Pitch[2] + z * dst2Pitch[3];
            tOutput2[dst2Addr] =argmaxNHWC;
            
          }
        } // W
      }   // H
    }     // C
  }       // N
}

template <typename srcType, typename dstType>
inline void fwdLibMaxPoolInstThreaded(
    bool argMax, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches, void *srcMatrixPitchesNoPadding,
    void *activations, void *activationsDims, void *activationsPitches,
    void *pkernels, void *pstrides, void *ppads, const float *scale, const int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<dstType> tOutput(dstMatrix, scale[1], offset[1]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  int64_t *tOutput2 = (int64_t *)dst2Matrix;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *dst2Pitch = (unsigned int *)dst2MatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *srcPitchNoPadding = (unsigned int *)  srcMatrixPitchesNoPadding;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides;
  unsigned int *pads = (unsigned int *)ppads;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[4] = {0, 0, 0, 0};
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 4, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y;
  while (!done && (offsetOut < posMax)) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);

    bool first = true;
    auto max_value = tAInput[0];
    max_value = 0;
    int64_t argmaxNHWC = 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {
      for (size_t fy = 0; fy < kernels[1]; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }

        int64_t idx = coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
          (size_t)oy * actPitch[2] + coord[3] * actPitch[3];
        auto val = tAInput[idx];
        if (first || (val >= max_value)) {
          first = false;
          max_value = val;
          if (argMax) 
            argmaxNHWC = coord[0] * srcPitchNoPadding[0] +
              (size_t)ox * srcPitchNoPadding[1] +
              (size_t)oy * srcPitchNoPadding[2]
              + coord[3];
        }
      }
    }

    tOutput[offsetOut] = max_value;

    if (argMax) {
      int64_t dst2Addr = coord[0] * dst2Pitch[0] + coord[1] * dst2Pitch[1] +
                         coord[2] * dst2Pitch[2] + coord[3] * dst2Pitch[3];
      tOutput2[dst2Addr] = argmaxNHWC;
    }
    done = getOffsets(4, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _MAX_POOL_INST_H_
