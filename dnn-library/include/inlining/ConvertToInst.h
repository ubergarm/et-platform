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

#ifndef _CONVERT_TO_INST_H_
#define _CONVERT_TO_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "utils.h" // From include/internal path
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind srcElK, ElemKind dstElK, bool alignedSrc, bool alignedDst>
inline void loadConvertStore(uintptr_t srcAddr, uintptr_t dstAddr, uint64_t conf, float indices, float indicesHigh,
                             uint64_t dstConf, float dstIndices, float dstIndicesHigh, float srcScale, float srcOffset,
                             float dstScaleReciprocal, float dstOffset) {

  constexpr size_t srcBytesPerElement = Type::getElementSize(srcElK);
  constexpr size_t dstBytesPerElement = Type::getElementSize(dstElK);
  constexpr bool sameConfig = isSameConfig<srcElK, dstElK, alignedSrc, alignedDst>();

  float op0 = 0.f, op0High = 0.f;
  load<srcBytesPerElement, alignedSrc>(srcAddr, conf, indices, indicesHigh, op0, op0High);

  float op1 = 0.f, op1High = 0.f;
  convert<srcElK, dstElK>(op0, op0High, op1, op1High, srcScale, srcOffset, dstScaleReciprocal, dstOffset);

  if constexpr (sameConfig) {
    store<dstBytesPerElement, alignedDst>(dstAddr, conf, indices, indicesHigh, op1, op1High);
  } else {
    store<dstBytesPerElement, alignedDst>(dstAddr, dstConf, dstIndices, dstIndicesHigh, op1, op1High);
  }
}

template <ElemKind dstElK, ElemKind srcElK>
inline __attribute__((always_inline)) void
fwdLibConvertToInstVectorized(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset = 0,
                              const uint32_t assignedMinions = 0) {

  constexpr size_t srcBytesPerElement = Type::getElementSize(srcElK);
  constexpr size_t dstBytesPerElement = Type::getElementSize(dstElK);
  constexpr bool alignedSrc = false;
  constexpr bool alignedDst = false;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;

  if (minionId >= activeMinions) {
    return;
  }

  void* srcT = inT->getRawDataPointer<void>();
  void* dstT = outT->getRawDataPointer<void>();

  const dim_t* srcIndex = inT->dims().data();
  const dim_t* srcPitch = inT->strides().data();
  const dim_t* dstPitch = outT->strides().data();

  size_t srcDimNum = inT->ndims();

  // Total number of elements in the tensor
  unsigned int numElemsDst = dstPitch[0] * srcIndex[0];

  // Each minion does a region of maxRead consecutive elements starting at
  // initialAddr
  unsigned int initialAddr, maxRead;
  getCachelinePartition(dstBytesPerElement, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dstT);

  if (maxRead == 0) {
    return;
  }

  // Destination tensor coordinates
  unsigned int coord[srcDimNum];

  // Number of non-zero coordinates
  unsigned int k = 0;

  // We move the initialAddr to the next non-padding position
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, srcIndex, k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += srcPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n");

  uint64_t conf;
  float indices;
  float indicesHigh;
  uint64_t dstConf;
  float dstIndices;
  float dstIndicesHigh;
  setupGatherScatterConfig<srcElK, dstElK, false, false>(conf, indices, indicesHigh, dstConf, dstIndices, dstIndicesHigh);

  float srcScale, srcOffset;
  float srcScaleScalar = inT->getScale();
  int32_t srcOffsetScalar = outT->getOffset();
  (void)srcScale;
  (void)srcOffset;
  (void)srcScaleScalar;
  (void)srcOffsetScalar;
  if constexpr (isQuantizedElemKind(srcElK)) {
    setupDequantize(srcScale, srcOffset, srcScaleScalar, srcOffsetScalar);
  }

  float dstScaleReciprocal, dstOffset;
  float dstScaleScalar = outT->getScale();
  int32_t dstOffsetScalar = outT->getOffset();
  (void)dstScaleReciprocal;
  (void)dstOffset;
  (void)dstScaleScalar;
  (void)dstOffsetScalar;
  if constexpr (isQuantizedElemKind(srcElK)) {
    setupQuantize(dstScaleReciprocal, dstOffset, dstScaleScalar, dstOffsetScalar);
  }

  unsigned int posMax = maxRead + initialAddr;

  bool done = false;
  while (not done and offsetOut < posMax) {
    uintptr_t srcAddr = (uintptr_t)srcT + offsetIn * srcBytesPerElement;
    uintptr_t dstAddr = (uintptr_t)dstT + offsetOut * dstBytesPerElement;
    loadConvertStore<srcElK, dstElK, alignedSrc, alignedDst>(srcAddr, dstAddr, conf, indices, indicesHigh, dstConf,
                                                             dstIndices, dstIndicesHigh, srcScale, srcOffset,
                                                             dstScaleReciprocal, dstOffset);
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, srcIndex, srcPitch, dstPitch);
  }

  if (DO_EVICTS) {
    unsigned int clperminion = maxRead * dstBytesPerElement / CACHE_LINE_BYTES;
    if (clperminion > 0) {
      evict_va_multi(DO_EVICTS, (uintptr_t)dstT + dstBytesPerElement * initialAddr, clperminion);
    }
  }
}

template <ElemKind dstElK, ElemKind srcElK>
inline __attribute__((always_inline)) void fwdLibConvertToInst(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                                               const uint32_t minionOffset = 0,
                                                               const uint32_t assignedMinions = 0) {
  dnn_lib::inlining::fwdLibConvertToInstVectorized<dstElK, srcElK>(outT, inT, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _CONVERT_TO_INST_H_
