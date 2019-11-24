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
void dnn_lib::fwdLibAvgPoolInst(void *dstMatrix, void *dstMatrixDims,
                                void *dstMatrixPitches, void *activations,
                                void *activationsDims, void *activationsPitches,
                                void *pkernels, void *pstrides, void *ppads,
                                float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides;
  unsigned int *pads = (unsigned int *)ppads;

  float filterArea = kernels[0] * kernels[1];
  float invFilter;
  fpReciprocalSingleElement(filterArea, invFilter);

  // For each input in the batch:
  for (size_t n = 0; n < dstIndex[0]; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < actIndex[3]; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pads[0]);
      for (size_t ax = 0; ax < dstIndex[1]; x += strides[0], ax++) {
        ssize_t y = -ssize_t(pads[1]);
        for (size_t ay = 0; ay < dstIndex[2]; y += strides[1], ay++) {
          auto sum = tAInput[0];
          sum = 0;

          for (size_t fx = 0; fx < kernels[0]; fx++) {
            for (size_t fy = 0; fy < kernels[1]; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
                  oy >= ssize_t(actIndex[2])) {
                continue;
              }

              sum += tAInput[n * actPitch[0] + (size_t)ox * actPitch[1] +
                             (size_t)oy * actPitch[2] + z];
            }
          }
          float tmp = sum;
          tmp *= invFilter;
          sum = tmp;
          tOutput[n * dstPitch[0] + (size_t)ax * dstPitch[1] +
                  (size_t)ay * dstPitch[2] + z] = sum;
        } // W
      }   // H
    }     // C
  }       // N
}

// template <typename srcType>
// void dnn_lib::fwdLibAvgPoolInst_Copy(void *dstMatrix, void *dstMatrixDims,
//                                      void *dstMatrixPitches, void *activations,
//                                      void *activationsDims, void
//                                      *activationsPitches, void *pkernels, void
//                                      *pstrides, void *ppads, float *scale, int32_t
//                                      *offset) {
//  Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
//  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
//
//  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
//  unsigned int *actIndex = (unsigned int *)activationsDims;
//
//  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
//  unsigned int *actPitch = (unsigned int *)activationsPitches;
//
//  unsigned int *kernels = (unsigned int *)pkernels;
//  unsigned int *strides = (unsigned int *)pstrides;
//  unsigned int *pads = (unsigned int *)ppads;
//
//  float filterArea = kernels[0] * kernels[1];
//  float invFilter;
//  fpReciprocalSingleElement(filterArea, invFilter);
//  unsigned int minionId = get_minion_id();
//  unsigned int numElemsKernel = kernels[0]*kernels[1];
//  unsigned int minionsperkernel = 1;
//  int level = -1;
//  while (minionsperkernel < numElemsKernel) {
//    minionsperkernel*= 2;
//    ++level;
//  }
//  unsigned int numKernels = activeMinions/minionsperkernel;
//  unsigned int kernel_id = minionId/minionsperkernel;
//  unsigned int kernel_minionId = minionId - kernel_id*minionsperkernel;
//  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];
//  unsigned int cll = 64/sizeof(srcType);
//  unsigned int ncl = (numElemsDst - 1)/cll + 1; //Amount of cache lines
//  unsigned int kcl = (ncl-1)/numKernels + 1; //Amount of cache lines to do for
//  each kernel unsigned int initialAddr = kcl*cll*kernel_id; unsigned int
//  maxRead = kcl*cll; unsigned int posMax = maxRead + initialAddr;
//
//  if (initialAddr >= numElemsDst) return;
//
//  unsigned int offsetOut = initialAddr;
//
//  long unsigned int coord[4] = {0,0,0,0};
//  unsigned int rm = initialAddr;
//  for (unsigned int i = 0; i < 4; i++) {
//    coord[i] = rm/dstPitch[i];
//    rm = rm-coord[i]*dstPitch[i];
//  }
//
//  unsigned int k = 4; //If it is a padding position we compute next useful
//  position for (unsigned int j = 3; j > 0; j--) {
//    if (coord[j] >= dstIndex[j]) {
//      coord[j-1]++;
//      k = j;
//    }
//  }
//  for (unsigned int j = k; j < 4; j++) coord[j] = 0;
//
//  bool done = false;
//  ssize_t dx, dy, x, y;
//  dx = kernel_minionId/kernels[1] - ssize_t(pads[0]);
//  dy = kernel_minionId%kernels[1] - ssize_t(pads[1]);
//  while(!done) {
//
//    x = coord[1]*strides[0] + dx;
//    y = coord[2]*strides[1] + dy;
//
//    auto sum = tAInput[0];
//    sum = 0;
//    if (x >= 0 && y >= 0 && x < ssize_t(actIndex[1]) &&
//        y < ssize_t(actIndex[2]) && kernel_minionId < numElemsKernel) {
//      sum = tAInput[coord[0]*actPitch[0] + x*actPitch[1] + y*actPitch[2] +
//      coord[3]*actPitch[3]];
//    }
//
//    for (int i = 0; i <= level; i++) {
//      sum = tensor_reduce_float(sum, 0x0, 1, i, 0x3);
//    }
//
//    if (kernel_minionId == 0) {
//      int64_t dstAddr = coord[0]*dstPitch[0] + coord[1]*dstPitch[1] +
//                         coord[2]*dstPitch[2] + coord[3]*dstPitch[3];
//      float tmp = sum;
//      tmp *= invFilter;
//      sum = tmp;
//      tOutput[dstAddr] = sum;
//    }
//
//    for (int j = 3; j >= 0; j--) {
//      if (coord[j] != (dstIndex[j] - 1)) {
//        offsetOut += dstPitch[j];
//        coord[j]++;
//        break;
//      } else if (j == 0) {
//        done = true;
//        break;
//      } else {
//        offsetOut -= (dstIndex[j] - 1) * dstPitch[j];
//        coord[j] = 0;
//      }
//    }
//    if (offsetOut >= posMax) break;
//
//  }
//}

template <typename srcType, typename dstType>
void dnn_lib::fwdLibAvgPoolInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *pkernels, void *pstrides, void *ppads, float *scale, int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<dstType> tOutput(dstMatrix, scale[1], offset[1]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides;
  unsigned int *pads = (unsigned int *)ppads;

  float filterArea = kernels[0] * kernels[1];
  float invFilter;
  fpReciprocalSingleElement(filterArea, invFilter);

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
  for (int i = 0; i < k; i++) {
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

    auto sum = tAInput[0];
    sum = 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {
      for (size_t fy = 0; fy < kernels[1]; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }

        sum += tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                       (size_t)oy * actPitch[2] + coord[3] * actPitch[3]];
      }
    }

    float tmp = sum;
    tmp *= invFilter;
    sum = tmp;
    tOutput[offsetOut] = sum;

    done = getOffsets(4, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

GEN_INSTANCES_OP(template, fwdLibAvgPoolInst,void *dstMatrix, void *dstMatrixDims,
                 void *dstMatrixPitches, void *activations,
                 void *activationsDims, void *activationsPitches,
                 void *pkernels, void *pstrides, void *ppads,
                 float *scale, int32_t *offset);

GEN_INSTANCES_2TYPE_OP(template, fwdLibAvgPoolInstThreaded,void *dstMatrix, void *dstMatrixDims,
                         void *dstMatrixPitches, void *activations,
                         void *activationsDims, void *activationsPitches,
                         void *pkernels, void *pstrides, void *ppads,
                         float *scale, int32_t *offset, uint64_t flags);
