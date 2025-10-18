/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _CONVERT_TO_INST_H_
#define _CONVERT_TO_INST_H_

#include "Compiler.h"
#include "LibTensor.h"
#include "LoadStore2.h"
#include "etsoc/common/utils.h"
#include "utils.h"
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

template <ElemKind srcElK, ElemKind dstElK, bool alignedSrc, bool alignedDst>
INLINE_ATTR void loadConvertStore(const uintptr_t dstAddr, const uintptr_t srcAddr, const dim_t valid,
                                  const float& srcScaleScalar, const int32_t& srcOffsetScalar,
                                  const float& dstScaleScalar, const int32_t& dstOffsetScalar) {
#if COMPILER_GCC
  __asm__ __volatile__("mov.m.x m0, zero, 0xFF \n");
#endif

  constexpr size_t srcBytesPerElement = Type::getElementSize(srcElK);
  constexpr size_t dstBytesPerElement = Type::getElementSize(dstElK);

  uint64_t conf;
  dnn_lib_v2::v8s32_t indices;
  dnn_lib_v2::v8s32_t indicesHigh;
  uint64_t dstConf;
  dnn_lib_v2::v8s32_t dstIndices;
  dnn_lib_v2::v8s32_t dstIndicesHigh;
  dnn_lib_v2::setupGatherScatterConfig<srcBytesPerElement, dstBytesPerElement, false, false>(
    conf, indices, indicesHigh, dstConf, dstIndices, dstIndicesHigh);
  dnn_lib_v2::v8f32_t srcScale;
  dnn_lib_v2::v8s32_t srcOffset;
  if constexpr (isQuantizedElemKind(srcElK)) {
    dnn_lib_v2::setupDequantize(srcScale, srcOffset, srcScaleScalar, srcOffsetScalar);
  }

  dnn_lib_v2::v8f32_t dstScaleReciprocal;
  dnn_lib_v2::v8s32_t dstOffset;
  if constexpr (isQuantizedElemKind(srcElK)) {
    dnn_lib_v2::setupQuantize(dstScaleReciprocal, dstOffset, dstScaleScalar, dstOffsetScalar);
  }

  // Enable only the valid elements
  et_assert(valid <= 8);
  uint8_t vmask = static_cast<uint8_t>(((1UL << valid) - 1));
#if COMPILER_GCC
  __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(vmask) :);
#endif

  constexpr bool sameConfig = isSameConfig<srcBytesPerElement, dstBytesPerElement, alignedSrc, alignedDst>();

  dnn_lib_v2::v8u32_t op0{0};
  dnn_lib_v2::v8u32_t op0High{0};
  dnn_lib_v2::load<dnn_lib_v2::v8u32_t, srcBytesPerElement, alignedSrc>(srcAddr, conf, indices, indicesHigh, op0,
                                                                        op0High, vmask);
  dnn_lib_v2::v8u32_t op1{0};
  dnn_lib_v2::v8u32_t op1High{0};
  dnn_lib_v2::convert<dnn_lib_v2::v8u32_t, srcElK, dnn_lib_v2::v8u32_t, dstElK>(
    op0, op0High, op1, op1High, srcScale, srcOffset, dstScaleReciprocal, dstOffset, vmask);

  if constexpr (sameConfig) {
    dnn_lib_v2::store<dnn_lib_v2::v8u32_t, dstBytesPerElement, alignedDst>(dstAddr, conf, indices, indicesHigh, op1,
                                                                           op1High, vmask);
  } else {
    dnn_lib_v2::store<dnn_lib_v2::v8u32_t, dstBytesPerElement, alignedDst>(dstAddr, dstConf, dstIndices, dstIndicesHigh,
                                                                           op1, op1High, vmask);
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

  float dstScaleScalar = outT->getScale();
  int32_t dstOffsetScalar = outT->getOffset();

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
