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

#ifndef _CONVOLUTION_3D_INST_H_
#define _CONVOLUTION_3D_INST_H_

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

  template <ElemKind elK, size_t N>
inline void fwdLibConvolution3DInst(LibTensor* outT, LibTensor* in1T,
                                    LibTensor* in2T, LibTensor* in3T,
                                    const std::array<uint32_t, N> &kernels,
                                    const std::array<uint32_t, N> &strides,
                                    const std::array<uint32_t, N> &pads,
                                    unsigned int group,
                                    uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  if (get_minion_id() != minionOffset) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT->dest in1T->activations in2T-> weight in3T->bias */
  void *dstMatrix = outT->getRawDataPointer<void>();
  void *activations = in1T->getRawDataPointer<void>();
  void *weights = in2T->getRawDataPointer<void>();
  
  // Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  Addresser<srcType> tOutput(dstMatrix, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tAInput(activations, in1T->getScale(), in1T->getOffset());
  // const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  const Addresser<srcType> tWInput(weights, in2T->getScale(), in2T->getOffset());
  // float *tBias = (float *)bias;
  float* tBias = in3T->getRawDataPointer<float>();

  // unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const dim_t *actIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const dim_t *actPitch = in1T->strides().data();
  // unsigned int *weightPitch = (unsigned int *)weightPitches;
  const dim_t *weightPitch = in2T->strides().data();
  
  assert(actIndex[4] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[4] % group == 0 &&
         "Output channels must be divisible by group.");
  size_t inCperG = actIndex[4] / group;
  size_t outCperG = dstIndex[4] / group;

  // For each input in the batch:
  for (size_t n = 0; n < actIndex[0]; n++) {

    // For each group of input channels:
    for (size_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (size_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pads[0]);
        for (size_t ax = 0; ax < dstIndex[1]; x += strides[0], ax++) {
          ssize_t y = -ssize_t(pads[1]);
          for (size_t ay = 0; ay < dstIndex[2]; y += strides[1], ay++) {
            ssize_t z = -ssize_t(pads[2]);
            for (size_t az = 0; az < dstIndex[3]; z += strides[2], az++) {

              // For each element in the 3Dconvolution-filter:
              auto sum = tAInput[0];
              sum = 0;
              for (size_t fx = 0; fx < kernels[0]; fx++) {
                for (size_t fy = 0; fy < kernels[1]; fy++) {
                  for (size_t fz = 0; fz < kernels[2]; fz++) {
                    ssize_t ox = x + fx;
                    ssize_t oy = y + fy;
                    ssize_t oz = z + fz;

                    // Ignore index access below zero (this is due to padding).
                    if (ox < 0 || oy < 0 || oz < 0 ||
                        ox >= ssize_t(actIndex[1]) ||
                        oy >= ssize_t(actIndex[2]) ||
                        oz >= ssize_t(actIndex[3])) {
                      continue;
                    }
                    for (size_t fd = 0; fd < inCperG; fd++) {
                      auto op1 =
                          tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                                  fy * weightPitch[2] + fz * weightPitch[3] +
                                  fd];
                      auto op2 =
                          tAInput[n * actPitch[0] + (size_t)ox * actPitch[1] +
                                  (size_t)oy * actPitch[2] +
                                  (size_t)oz * actPitch[3] + g * inCperG + fd];
                      sum += op1 * op2;
                    }
                  }
                }
              }
              int64_t addr = n * dstPitch[0] + (size_t)ax * dstPitch[1] +
                             (size_t)ay * dstPitch[2] +
                             (size_t)az * dstPitch[3] + d;
              sum += tBias[d];
              tOutput[addr] = sum;
            } // D
          }   // W
        }     // H
      }       // C
    }         // G
  }           // N
}

  template <ElemKind elK, size_t N>
inline void fwdLibConvolution3DInstThreaded(LibTensor* outT, LibTensor* in1T,
                                            LibTensor* in2T, LibTensor* in3T,
                                            const std::array<uint32_t, N> &kernels,
                                            const std::array<uint32_t, N> &strides,
                                            const std::array<uint32_t, N> &pads,
                                            unsigned int group,
                                            uint64_t flags,
                                            const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  /* outT->dest in1T->activations in2T-> weight in3T->bias */

  void *dstMatrix = outT->getRawDataPointer<void>();
  void *activations = in1T->getRawDataPointer<void>();
  void *weights = in2T->getRawDataPointer<void>();
  
  // Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  Addresser<srcType> tOutput(dstMatrix, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tAInput(activations, in1T->getScale(), in1T->getOffset());
  // const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  const Addresser<srcType> tWInput(weights, in2T->getScale(), in2T->getOffset());
  // float *tBias = (float *)bias;
  float* tBias = in3T->getRawDataPointer<float>();
 
  // unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const dim_t *actIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const dim_t *actPitch = in1T->strides().data();
  // unsigned int *weightPitch = (unsigned int *)weightPitches;
  const dim_t *weightPitch = in2T->strides().data();
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;
  assert(actIndex[4] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[4] % group == 0 &&
         "Output channels must be divisible by group.");
  unsigned int inCperG = actIndex[4] / group;
  unsigned int outCperG = dstIndex[4] / group;

  // unsigned int eDstPitch[6] = {dstPitch[0], dstPitch[1], dstPitch[2],
  //                              dstPitch[3], outCperG,    1};
  // unsigned int eDstIndex[6] = {dstIndex[0], dstIndex[1], dstIndex[2],
  //                              dstIndex[3], group,       outCperG};
  size_t eDstPitch[6] = {dstPitch[0], dstPitch[1], dstPitch[2],
                               dstPitch[3], outCperG,    1};
  size_t eDstIndex[6] = {dstIndex[0], dstIndex[1], dstIndex[2],
                               dstIndex[3], group,       outCperG};

  unsigned int coord[6] = {0, 0, 0, 0, 0, 0};
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 6, eDstPitch, eDstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, z, d;
  while (!done && (offsetOut < posMax)) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);
    z = coord[3] * strides[2] - ssize_t(pads[2]);
    d = coord[4] * outCperG + coord[5];

    auto sum = tAInput[0];
    sum = 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {
      for (size_t fy = 0; fy < kernels[1]; fy++) {
        for (size_t fz = 0; fz < kernels[2]; fz++) {
          ssize_t ox = x + fx;
          ssize_t oy = y + fy;
          ssize_t oz = z + fz;

          // Ignore index access below zero (this is due to padding).
          if (ox < 0 || oy < 0 || oz < 0 || ox >= ssize_t(actIndex[1]) ||
              oy >= ssize_t(actIndex[2]) || oz >= ssize_t(actIndex[3])) {
            continue;
          }
          for (size_t fd = 0; fd < inCperG; fd++) {
            auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                               fy * weightPitch[2] + fz * weightPitch[3] + fd];
            auto op2 =
                tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                        (size_t)oy * actPitch[2] + (size_t)oz * actPitch[3] +
                        coord[4] * inCperG + fd];
            sum += op1 * op2;
          }
        }
      }
    }
    sum += tBias[d];
    tOutput[offsetOut] = sum;

    done = getOffsets(6, coord, offsetOut, eDstIndex, eDstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _CONVOLUTION_3D_INST_H_
