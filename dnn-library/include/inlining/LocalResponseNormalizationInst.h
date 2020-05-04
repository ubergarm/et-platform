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

#ifndef _LOCAL_RESPONSE_NORMALIZATION_INST_H_
#define _LOCAL_RESPONSE_NORMALIZATION_INST_H_

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

template <typename srcType>
inline void fwdLibLocalResponseNormalizationInst(
    LibTensor* out1T, LibTensor* out2T, LibTensor* inT,
    unsigned int halfWindowSize, float alpha, float beta, float k) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  /* out1T --> dst  out2T--> dst2  inT--> data */

  void *dstMatrix = out1T->getRawDataPointer<void>();
  void *dst2Matrix = out2T->getRawDataPointer<void>();
  void *activations = inT->getRawDataPointer<void>();
  
  // Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
  Addresser<srcType> tOutput(dstMatrix, out1T->getScale(), out1T->getOffset());
  // Addresser<srcType> tScale(dst2Matrix, scale[2], offset[2]);
  Addresser<srcType> tScale(dst2Matrix, out2T->getScale(), out2T->getOffset());
  // const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tAInput(activations, inT->getScale(), inT->getOffset());  

  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const size_t *actIndex = inT->dims().data();  
  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const size_t *dstPitch = out1T->strides().data();
  // unsigned int *dst2Pitch = (unsigned int *)dst2MatrixPitches;
  const size_t *dst2Pitch = out2T->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const size_t *actPitch = inT->strides().data();

  

  // LRN node does not change the shape of the input.
  // assert((dstIndex[0] == actIndex[0]) && (dstIndex[1] == actIndex[1]) &&
  // (dstIndex[2] == actIndex[2]) && (dstIndex[3] == actIndex[3]) && "Output of
  // LRN node must be same shape as input");

  // LRN node normalizes across channels, so the input must have a minimum
  // depth of 1.
  // assert(actIndex[3] > 0 && "Input of LRN node must have a minimum depth of
  // 1");

  auto windowSize = 2 * halfWindowSize + 1;
  float inversedWindowSize;
  fpReciprocalSingleElement(windowSize, inversedWindowSize);
  float normedAlpha = alpha * inversedWindowSize;

  // For every input in the batch:
  for (size_t n = 0; n < actIndex[0]; n++) {

    // For every row:
    for (size_t h = 0; h < actIndex[1]; h++) {

      // For every column:
      for (size_t w = 0; w < actIndex[2]; w++) {

        // For every channel:
        for (size_t c = 0; c < actIndex[3]; c++) {
          auto squareSum = tAInput[0];
          squareSum = 0.0;
          for (size_t i = (c >= halfWindowSize ? c - halfWindowSize : 0);
               i <=
               std::min(c + halfWindowSize, (long unsigned int)actIndex[3] - 1);
               i++) {
            auto val = tAInput[n * actPitch[0] + h * actPitch[1] +
                               w * actPitch[2] + i * actPitch[3]];
            squareSum += val * val;
          }

          auto scale = k + normedAlpha * squareSum;

          // This will be used to accelerate the backward pass.
          tScale[n * dst2Pitch[0] + h * dst2Pitch[1] + w * dst2Pitch[2] +
                 c * dst2Pitch[3]] = scale;

          auto normFactor = getPow(scale, -beta);
          auto op = tAInput[n * actPitch[0] + h * actPitch[1] +
                            w * actPitch[2] + c * actPitch[3]];
          op *= normFactor;
          tOutput[n * dstPitch[0] + h * dstPitch[1] + w * dstPitch[2] +
                  c * dstPitch[3]] = op;
        }
      }
    }
  }
}

// First threaded version, assuming dstMatrix and dst2Matrix have the same
// Pitches. Without this assumption, coherence might be lost and therefore this
// version is not correct. Notice that dst2Matrix is only needed for backward
// pass, i.e. ETSOC won't be using it. Actually, we could skip generating it.

template <typename srcType>
inline void fwdLibLocalResponseNormalizationInstThreaded(LibTensor* out1T,
          LibTensor* out2T, LibTensor* inT, unsigned int halfWindowSize,
          float alpha, float beta, float k, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  /* out1T --> dst  out2T--> dst2  inT--> data */
  void *dstMatrix = out1T->getRawDataPointer<void>();
  void *dst2Matrix = out2T->getRawDataPointer<void>();
  void *activations = inT->getRawDataPointer<void>();

  // Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
  Addresser<srcType> tOutput(dstMatrix, out1T->getScale(), out1T->getOffset());
  // Addresser<srcType> tScale(dst2Matrix, scale[2], offset[2]);
  Addresser<srcType> tScale(dst2Matrix, out2T->getScale(), out2T->getOffset());
  // const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tAInput(activations, inT->getScale(), inT->getOffset());
  
  // unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const size_t *dstIndex = out1T->dims().data();
  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const size_t *actIndex = inT->dims().data(); 
  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const size_t *dstPitch = out1T->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const size_t *actPitch = inT->strides().data();
  
  // LRN node does not change the shape of the input.
  // assert((dstIndex[0] == actIndex[0]) && (dstIndex[1] == actIndex[1]) &&
  // (dstIndex[2] == actIndex[2]) && (dstIndex[3] == actIndex[3]) && "Output of
  // LRN node must be same shape as input");

  // LRN node normalizes across channels, so the input must have a minimum
  // depth of 1.
  // assert(actIndex[3] > 0 && "Input of LRN node must have a minimum depth of
  // 1");

  auto windowSize = 2 * halfWindowSize + 1;
  float inversedWindowSize;
  fpReciprocalSingleElement(windowSize, inversedWindowSize);
  float normedAlpha = alpha * inversedWindowSize;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  const unsigned int srcDimNum = 4;
  unsigned int coord[srcDimNum] = {0, 0, 0, 0};
  unsigned int t = 0;  //this variable is usually called k, but n this case the name k is already used
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           t);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < t; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    auto squareSum = tAInput[offsetIn];
    squareSum = 0.0;
    size_t c = size_t(coord[3]);
    for (unsigned int i = (c >= halfWindowSize ? c - halfWindowSize : 0);
         i <= std::min(c + halfWindowSize, (long unsigned int)actIndex[3] - 1);
         i++) {
      auto val = tAInput[offsetIn + size_t((i - c) * actPitch[3])];
      squareSum += val * val;
    }

    auto scale = k + normedAlpha * squareSum;

    // This will be used to accelerate the backward pass.
    tScale[offsetOut] = scale;

    auto normFactor = getPow(scale, -beta);
    auto op = tAInput[offsetIn];
    op *= normFactor;
    tOutput[offsetOut] = op;

    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, dstIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

template <typename srcType>
inline void fwdLibLocalResponseNormalizationInstVectorized(LibTensor* out1T,
    LibTensor* out2T, LibTensor* inT, unsigned int halfWindowSize, float alpha,
    float beta, float k, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  /* out1T --> dst  out2T--> dst2  inT--> data */

  void *dstMatrix = out1T->getRawDataPointer<void>();
  void *dst2Matrix = out2T->getRawDataPointer<void>();
  void *activations = inT->getRawDataPointer<void>();

  // uintptr_t srcAddr = (uintptr_t)activations;
  uintptr_t srcAddr = reinterpret_cast<uintptr_t>(activations);
  // Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
  Addresser<srcType> tOutput(dstMatrix, out1T->getScale(), out1T->getOffset());
  // Addresser<srcType> tScale(dst2Matrix, scale[2], offset[2]);
  Addresser<srcType> tScale(dst2Matrix, out2T->getScale(), out2T->getOffset());
  // const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tAInput(activations, inT->getScale(), inT->getOffset());
  
  // unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const size_t *dstIndex = out1T->dims().data();
  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const size_t *actIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const size_t *dstPitch = out1T->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const size_t *actPitch = inT->dims().data();

  auto windowSize = 2 * halfWindowSize + 1;
  float inversedWindowSize;
  fpReciprocalSingleElement(windowSize, inversedWindowSize);
  float normedAlpha = alpha * inversedWindowSize;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  const unsigned int srcDimNum = 4;
  unsigned int coord[srcDimNum] = {0, 0, 0, 0};
  unsigned int t = 0;  //this variable is usually called k, but n this case the name k is already used
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           t);
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < t; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int mask;
  while (!done && (offsetOut < posMax)) {
    float squareSum = 0.0;
    size_t c = size_t(coord[3]);
    unsigned int start = (c >= halfWindowSize ? c - halfWindowSize : 0);
    unsigned int end = std::min(c + halfWindowSize, (long unsigned int)actIndex[3] - 1);
    unsigned int registers = (end - start + 1)/8;
    unsigned int mod = (end - start + 1) - 8*registers;
    constexpr uint32_t offs = 32;
    srcAddr += (offsetIn + (start - c)*actPitch[3]) * typeSize;

    mask = ((1 << mod) - 1);
    __asm__ __volatile__("mov.m.x m1, %[mask], 0 \n"
                         "mov.m.x m0, zero, 0xff \n"
                         "fxor.pi f0, f0, f0\n"
                         "add t0, zero, zero\n"


                         "ble %[registers], t0, 2f\n"
                         "1:\n"
                         "flw.ps f1, 0x0(%[src])\n"
                         "fmul.ps f1, f1, f1\n"
                         "fadd.ps f0, f0, f1\n"
                         "addi t0, t0, 0x1\n"
                         "addi %[src], %[src], %[offs]\n"
                         "blt t0, %[registers], 1b\n"
                         "2:\n"
                         "ble %[mod], zero, 3f\n"
                         "maskand m0, m1, m0 \n"
                         "flw.ps f1, 0x0(%[src])\n"
                         "fmul.ps f1, f1, f1\n"
                         "fadd.ps f0, f0, f1\n"
                         "3:\n"
                         "mov.m.x m0, zero, 0xff \n"
                         "fswizz.ps f30, f0, 0xe \n"
                         "fadd.ps f0,f30, f0 \n"
                         "fswizz.ps f30, f0, 0x1 \n"
                         "fadd.ps f0,f30, f0 \n"
                         "fmvs.x.ps %[sum], f0, 0x4 \n"
                         "fmv.w.x f30, %[sum] \n"
                         "fadd.s f0, f30, f0 \n"
                         "fmvs.x.ps %[sum], f0, 0x0 \n"

                         : [ sum ] "=r"(squareSum),
                           [ src ] "+&r"(srcAddr)
                         : [ mask ] "r"(mask),
                           [ mod ] "r"(mod),
                           [ offs ] "I"(offs),
                           [ registers ] "r"(registers)

                         : "t0", "f0", "f1", "f30", "memory");

    auto scale = k + normedAlpha * squareSum;

    tScale[offsetOut] = scale;

    auto normFactor = getPow(scale, -beta);
    auto op = tAInput[offsetIn];
    op *= normFactor;
    tOutput[offsetOut] = op;



    srcAddr = (uintptr_t)activations;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _LOCAL_RESPONSE_NORMALIZATION_INST_H_
