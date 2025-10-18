/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _LOCAL_RESPONSE_NORMALIZATION_INST_H_
#define _LOCAL_RESPONSE_NORMALIZATION_INST_H_

#include "Addresser.h"
#include "Float16.h"
#include "LibTensor.h"
#include "LoadStore.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
INLINE_ATTR void fwdLibLocalResponseNormalizationInst(LibTensor* out1T, LibTensor* out2T, LibTensor* inT,
                                                      unsigned int halfWindowSize, float alpha, float beta, float k,
                                                      uint64_t flags, const uint32_t minionOffset = 0,
                                                      const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* out1T --> dst  out2T--> dst2  inT--> data */
  void* dstMatrix = out1T->getRawDataPointer();
  void* dst2Matrix = out2T->getRawDataPointer();
  void* activations = inT->getRawDataPointer();

  Addresser<elK> tOutput(dstMatrix, out1T->getScale(), out1T->getOffset());
  Addresser<elK> tScale(dst2Matrix, out2T->getScale(), out2T->getOffset());
  const Addresser<elK> tAInput(activations, inT->getScale(), inT->getOffset());
  
  const dim_t *dstIndex = out1T->dims().data();
  const dim_t *actIndex = inT->dims().data(); 
  const dim_t *dstPitch = out1T->strides().data();
  const dim_t* dstPitch2 = out2T->strides().data();
  const dim_t *actPitch = inT->strides().data();

  // Input and output dimensions should match
  assert(dstIndex[0] == actIndex[0] and dstIndex[1] == actIndex[1] and dstIndex[2] == actIndex[2] and
         dstIndex[3] == actIndex[3]);

  auto windowSize = static_cast<float>(2 * halfWindowSize + 1);
  float inversedWindowSize;
  fpReciprocalSingleElement(windowSize, inversedWindowSize);
  float normedAlpha = alpha * inversedWindowSize;

  size_t numElemsDst = dstPitch[0] * dstIndex[0];

  size_t initialAddr, maxRead;
  constexpr size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstMatrix);
  if (maxRead == 0)
    return;

  const size_t srcDimNum = 4;
  dim_array_t coord = {0};
  dim_t numDimensions = 0;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex, numDimensions);

  size_t offsetIn = 0;
  size_t offsetOut = 0;
  size_t offsetOut2 = 0;
  for (dim_t j = 0; j < numDimensions; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
    offsetOut2 += dstPitch2[j] * coord[j];
  }

  size_t posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {

    size_t c = coord[3];
    size_t dim = actIndex[3];
    size_t pitch = actPitch[3];

    size_t startCoord = (c >= halfWindowSize) ? (c - halfWindowSize) : 0;
    size_t endCoord = std::min(c + halfWindowSize, dim - 1);

    size_t startAddr = offsetIn + (startCoord - c) * pitch;
    size_t endAddr = offsetIn + (endCoord - c) * pitch;

    auto squareSum = tAInput[offsetIn];
    squareSum = 0.0;

    for (size_t srcAddr = startAddr; srcAddr <= endAddr; srcAddr += pitch) {
      auto val = tAInput[srcAddr];
      squareSum += val * val;
    }

    auto scale = k + normedAlpha * squareSum;

    // This will be used to accelerate the backward pass.
    tScale[offsetOut2] = scale;

    auto normFactor = getPow(scale, -beta);
    auto op = tAInput[offsetIn];
    op *= normFactor;
    tOutput[offsetOut] = op;

    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, offsetOut2, dstIndex, actPitch, dstPitch, dstPitch2);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

/* Reduce a vector by means of the addition */
static INLINE_ATTR void reduceAdd(float value, float& result) {
  // +----+-------+----+-------------+----+-------+----+-------------+-------------------------------------+
  // | h3 |  h2   | h1 |     h0      | l3 |  l2   | l1 |     l0      | value                               |
  // +----+-------+----+-------------+----+-------+----+-------------+-------------------------------------+
  // | -  |  h3   | -  |     h1      | -  |  l3   | -  |     l1      | 1) temp <= swizzle(value, "x3x1")   |
  // +----+-------+----+-------------+----+-------+----+-------------+-------------------------------------+
  // | -  | h2+h3 | -  |    h0+h1    | -  | l2+l3 | -  |    l0+h1    | 2) temp2 <= value + temp            |
  // +----+-------+----+-------------+----+-------+----+-------------+-------------------------------------+
  // | -  |   -   | -  |    h2+h3    | -  |   -   | -  |    l2+h3    | 3) temp <= swizzle(temp2, "xxx2")   |
  // +----+-------+----+-------------+----+-------+----+-------------+-------------------------------------+
  // | -  |   -   | -  | h0+h1+h2+h3 | -  |   -   | -  | l0+h1+l2+h3 | 4) temp2 <= temp2 + temp            |
  // +----+-------+----+-------------+----+-------+----+-------------+-------------------------------------+
  // | -  |   -   | -  |      -      | -  |   -   | -  | h0+h1+h2+h3 | 5) temp <= splat(extract(temp2, 4)) |
  // +----+-------+----+-------------+----+-------+----+-------------+-------------------------------------+
  // |    |       |    |             |    |       | -  | l0+h1+l2+h3 | 6) result <= temp2 + temp           |
  // | -  |   -   | -  |      -      | -  |   -   | -  | h0+h1+h2+h3 |                                     |
  // +----+-------+----+-------------+----+-------+----+-------------+-------------------------------------+
  float temp, temp2;
  uint64_t extract;
  __asm__ __volatile__("fswizz.ps %[temp], %[value], 0x31\n"    // 1)
                       "fadd.ps %[temp2], %[value], %[temp]\n"  // 2)
                       "fswizz.ps %[temp], %[temp2], 0x02\n"    // 3)
                       "fadd.ps %[temp2], %[temp2], %[temp]\n"  // 4)
                       "fmvs.x.ps %[extract], %[temp2], 0x4\n"  // 5)
                       "fbcx.ps %[temp], %[extract]\n"          //
                       "fadd.ps %[result], %[temp2], %[temp]\n" // 6)
                       : [ temp ] "=&f"(temp), [ temp2 ] "=f"(temp2), [ extract ] "=r"(extract), [ result ] "=f"(result)
                       : [ value ] "f"(value)
                       :);
}

template <ElemKind elK>
INLINE_ATTR void fwdLibLocalResponseNormalizationInstVectorized(LibTensor* out1T, LibTensor* out2T, LibTensor* inT,
                                                                unsigned int halfWindowSize, float alpha, float beta,
                                                                float k, uint64_t flags,
                                                                const uint32_t minionOffset = 0,
                                                                const uint32_t assignedMinions = 0) {

  using type = typename elemKind2elemTy<elK>::type;
  constexpr size_t bytesPerElement = getsize<type>();

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  void* dstMatrix = out1T->getRawDataPointer();
  void* dst2Matrix = out2T->getRawDataPointer();
  void* activations = inT->getRawDataPointer();

  const dim_t *dstIndex = out1T->dims().data();
  const dim_t *actIndex = inT->dims().data();
  const dim_t *dstPitch = out1T->strides().data();
  const dim_t* dstPitch2 = out2T->strides().data();
  const dim_t* actPitch = inT->strides().data();

  // Input and output dimensions should match
  assert(dstIndex[0] == actIndex[0] and dstIndex[1] == actIndex[1] and dstIndex[2] == actIndex[2] and
         dstIndex[3] == actIndex[3]);

  auto windowSize = static_cast<float>(2 * halfWindowSize + 1);
  float inversedWindowSize;
  fpReciprocalSingleElement(windowSize, inversedWindowSize);
  float normedAlpha = alpha * inversedWindowSize;

  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  getCachelinePartition(bytesPerElement, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dstMatrix);

  if (maxRead == 0) {
    return;
  }

  const size_t srcDimNum = 4;
  dim_array_t coord = {0};
  dim_t numDimensions = 0;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex, numDimensions);

  size_t offsetIn = 0;
  size_t offsetOut = 0;
  size_t offsetOut2 = 0;
  for (dim_t j = 0; j < numDimensions; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
    offsetOut2 += dstPitch2[j] * coord[j];
  }

  __asm__ __volatile__("mov.m.x m0, zero, 0xff");

  // Setup gather config for source
  constexpr bool srcAligned = false;
  uint64_t srcConf;
  float srcIndices;
  setupGatherScatterConfig<bytesPerElement, srcAligned>(srcConf, srcIndices);

  // Setup scather config for activation destination tensor
  constexpr bool dstAligned = false;
  uint64_t dstConf;
  float dstIndices;
  setupGatherScatterConfig<bytesPerElement, dstAligned>(dstConf, dstIndices);

  // Setup scather config for scale destination tensor
  uint64_t dstConf2;
  float dstIndices2;
  setupGatherScatterConfig<bytesPerElement, dstAligned>(dstConf2, dstIndices2);

  size_t posMax = maxRead + initialAddr;
  bool done = false;
  while (not done and offsetOut < posMax) {

    size_t c = size_t(coord[3]);
    size_t start = (c >= halfWindowSize ? c - halfWindowSize : 0);
    size_t end = std::min(c + halfWindowSize, actIndex[3] - 1);
    size_t registers = (end - start + 1) / 8;
    size_t mod = (end - start + 1) - 8 * registers;

    uintptr_t srcAddr =
      reinterpret_cast<uintptr_t>(activations) + (offsetIn + (start - c) * actPitch[3]) * bytesPerElement;

    float value, squareSum, squareSumTail;
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n"
                         "fxor.pi %[squareSum], %[squareSum], %[squareSum]\n"
                         "fxor.pi %[squareSumTail], %[squareSumTail], %[squareSumTail]\n"
                         : [ squareSum ] "=f"(squareSum), [ squareSumTail ] "=f"(squareSumTail));

    constexpr uint32_t pitch = 8 * bytesPerElement;
    for (size_t i = 0; i < registers; ++i, srcAddr += pitch) {
      load<bytesPerElement, srcAligned>(srcAddr, srcConf, srcIndices, value);
      convert<elK, FloatTy>(value);
      __asm__ __volatile__("fmadd.ps %[squareSum], %[value], %[value], %[squareSum]\n"
                           : [ squareSum ] "+f"(squareSum)
                           : [ value ] "f"(value));
    }

    reduceAdd(squareSum, squareSum);

    if (mod > 0) {
      uint64_t mask = (1 << mod) - 1;
      __asm__ __volatile__("mov.m.x m0, %[mask], 0x0\n" : : [ mask ] "r"(mask));
      load<bytesPerElement, srcAligned>(srcAddr, srcConf, srcIndices, value);
      convert<elK, FloatTy>(value);
      __asm__ __volatile__("fmadd.ps %[squareSumTail], %[value], %[value], %[squareSumTail]\n"
                           : [ squareSumTail ] "+f"(squareSumTail)
                           : [ value ] "f"(value));
    }

    reduceAdd(squareSumTail, squareSumTail);

    float scale = k + normedAlpha * (squareSum + squareSumTail);

    // Load current element and convert to float if needed
    srcAddr = reinterpret_cast<uintptr_t>(activations) + offsetIn * bytesPerElement;
    load<bytesPerElement, srcAligned>(srcAddr, srcConf, srcIndices, value);
    convert<elK, FloatTy>(value);

    // Multiply the normalization factor
    float output = value * getPow(scale, -beta);

    // Convert to elK and store
    convert<FloatTy, elK>(output);
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dstMatrix) + offsetOut * bytesPerElement;
    store<bytesPerElement, dstAligned>(dstAddr, dstConf, dstIndices, output);

    // Convert scale to elK and store
    convert<FloatTy, elK>(scale);
    uintptr_t dstAddr2 = reinterpret_cast<uintptr_t>(dst2Matrix) + offsetOut2 * bytesPerElement;
    store<bytesPerElement, dstAligned>(dstAddr2, dstConf2, dstIndices2, scale);

    // Move to next element
    // FIXME: untouchable dest and scale padding is touched in some dimensoins. --src-dims="1,39,3,2"
    // see SW-10757. for now ImplSelector selects scalar implementation in touchable padding cases.
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, offsetOut2, actIndex, actPitch, dstPitch, dstPitch2);
  }

  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * bytesPerElement + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0)
    evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + bytesPerElement * initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _LOCAL_RESPONSE_NORMALIZATION_INST_H_
