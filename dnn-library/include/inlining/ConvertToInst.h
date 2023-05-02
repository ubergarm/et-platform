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
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

#include "LibTensor.h"
#include "LoadStore.h"
#include "utils.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind srcElK, ElemKind dstElK, bool alignedSrc, bool alignedDst>
INLINE_ATTR void loadConvertStore(const uintptr_t dstAddr, const uintptr_t srcAddr, const dim_t valid,
                                  const float& srcScaleScalar, const int32_t& srcOffsetScalar,
                                  const float& dstScaleScalar, const int32_t& dstOffsetScalar) {
  __asm__ __volatile__("mov.m.x m0, zero, 0xFF \n");

  constexpr size_t srcBytesPerElement = Type::getElementSize(srcElK);
  constexpr size_t dstBytesPerElement = Type::getElementSize(dstElK);

  uint64_t conf;
  float indices;
  float indicesHigh;
  uint64_t dstConf;
  float dstIndices;
  float dstIndicesHigh;
  setupGatherScatterConfig<srcBytesPerElement, dstBytesPerElement, false, false>(conf, indices, indicesHigh, dstConf,
                                                                                 dstIndices, dstIndicesHigh);
  float srcScale, srcOffset;
  // (void)srcScale;
  // (void)srcOffset;
  if constexpr (isQuantizedElemKind(srcElK)) {
    setupDequantize(srcScale, srcOffset, srcScaleScalar, srcOffsetScalar);
  }
  float dstScaleReciprocal, dstOffset;
  // (void)dstScaleReciprocal;
  // (void)dstOffset;
  if constexpr (isQuantizedElemKind(srcElK)) {
    setupQuantize(dstScaleReciprocal, dstOffset, dstScaleScalar, dstOffsetScalar);
  }

  // Enables only the valid elements
  if (valid < 8) {
    uint8_t mask = static_cast<uint8_t>(((1UL << valid) - 1));
    __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
  } else {
    __asm__ __volatile__("mov.m.x m0, zero, 0xFF \n");
  }

  constexpr bool sameConfig = isSameConfig<srcBytesPerElement, dstBytesPerElement, alignedSrc, alignedDst>();

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

  using dstType = typename elemKind2elemTy<dstElK>::type;
  using srcType = typename elemKind2elemTy<srcElK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions =
    (assignedMinions == 0) ? static_cast<uint32_t>(MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  constexpr bool alignedSrc = false;
  constexpr bool alignedDst = false;

  float srcScaleScalar = inT->getScale();
  int32_t srcOffsetScalar = inT->getOffset();
  // (void)srcScaleScalar;
  // (void)srcOffsetScalar;

  float dstScaleScalar = outT->getScale();
  int32_t dstOffsetScalar = outT->getOffset();
  // (void)dstScaleScalar;
  // (void)dstOffsetScalar;

  outT->partitionLoop<dstType, srcType>(minionId, activeMinions, flags, inT,
                                        loadConvertStore<srcElK, dstElK, alignedSrc, alignedDst>, srcScaleScalar,
                                        srcOffsetScalar, dstScaleScalar, dstOffsetScalar);
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
