/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef __MAX_SPLAT_INST_H_
#define __MAX_SPLAT_INST_H_

#include "Float16.h"
#include "LibTensor.h"
#include "LoadStore2.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>
#include <tuple>

namespace dnn_lib {

namespace inlining {

template <ElemKind elK, bool srcAligned, bool dstAligned>
INLINE_ATTR void
maxSplatOp(const uintptr_t dst, [[maybe_unused]] const uintptr_t src, [[maybe_unused]] const dim_t valid,
           [[maybe_unused]] uint64_t srcConf, [[maybe_unused]] dnn_lib_v2::v8s32_t srcIndices,
           [[maybe_unused]] dnn_lib_v2::v8s32_t srcIndicesHigh, [[maybe_unused]] uint64_t dstConf,
           [[maybe_unused]] dnn_lib_v2::v8s32_t dstIndices, [[maybe_unused]] dnn_lib_v2::v8s32_t dstIndicesHigh,
           dnn_lib_v2::v8u32_t splatLow, [[maybe_unused]] dnn_lib_v2::v8u32_t splatHigh) {

  constexpr size_t bytesPerElement = Type::getElementSize(elK);

  // Whether it makes sense to calculate the maximum of two values
  // with the same scale and offset as it would be done by
  // std::min(static_cast<int32_t>(a), static_cast<int32_t>(b)).
  constexpr bool isAKnownSuitableForMaxInt32 = elK == Int8QTy or elK == Int16QTy or elK == Int32QTy or elK == Int32ITy;

  // Whether it makes sense to calculate the maximum of two values
  // with the same scale and offset as it would be done by
  // std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b)).
  constexpr bool isAKnownSuitableForMaxUInt32 = elK == UInt8QTy or elK == UInt8FusedQTy or elK == UInt8FusedFP16QTy or
                                                elK == UInt4FusedFP16QTy or elK == UInt4FusedQTy or elK == BoolTy;

  // Whether it makes sense to calculate the maximum of two values
  // with the same scale and offset as it would be done by
  // std::min(static_cast<float>(a), static_cast<float>(b)).
  constexpr bool isAKnownSuitableForMaxFloat = elK == FloatTy or elK == Float16Ty or elK == BFloat16Ty;

  // Whether it makes sense to calculate the maximum of two values
  // with the same scale and offset as it would be done by
  // std::min(static_cast<uint64_t>(a), static_cast<uint64_t>(b)).
  constexpr bool isAKnownSuitableForMaxInt64 = elK == Int64ITy;

  static_assert(isAKnownSuitableForMaxInt32 or isAKnownSuitableForMaxUInt32 or isAKnownSuitableForMaxFloat or
                isAKnownSuitableForMaxInt64);

  // Enable only the valid elements
  uint32_t mask = (1 << valid) - 1;
#if COMPILER_GCC
  __asm__ __volatile__("mov.m.x m0, %[maskValue], 0\n" : : [ maskValue ] "r"(mask) :);
#endif

  dnn_lib_v2::v8u32_t srcLow, srcHigh;
  dnn_lib_v2::load<dnn_lib_v2::v8u32_t, bytesPerElement, srcAligned>(src, srcConf, srcIndices, srcIndicesHigh, srcLow,
                                                                     srcHigh, mask);

  dnn_lib_v2::v8u32_t resultLow;
  dnn_lib_v2::v8u32_t resultHigh = {0};

  static_assert(isAKnownSuitableForMaxInt32 or isAKnownSuitableForMaxUInt32 or isAKnownSuitableForMaxFloat or
                isAKnownSuitableForMaxInt64);

  if constexpr (isAKnownSuitableForMaxInt32) {
    __asm__ __volatile__("fmax.pi %[resultLow], %[srcLow], %[splatLow]\n"
                         : [ resultLow ] "=f"(resultLow)
                         : FOR_CLANG([ vmask ] "M"(mask)) FOR_CLANG_COMMA[srcLow] "f"(srcLow),
                           [ splatLow ] "f"(splatLow));
  } else if constexpr (isAKnownSuitableForMaxUInt32) {
    __asm__ __volatile__("fmaxu.pi %[resultLow], %[srcLow], %[splatLow]\n"
                         : [ resultLow ] "=f"(resultLow)
                         : FOR_CLANG([ vmask ] "M"(mask)) FOR_CLANG_COMMA[srcLow] "f"(srcLow),
                           [ splatLow ] "f"(splatLow));
  } else if constexpr (isAKnownSuitableForMaxFloat) {
    if constexpr (elK != FloatTy) {
      dnn_lib_v2::convert<dnn_lib_v2::v8u32_t, elK, dnn_lib_v2::v8u32_t, FloatTy>(srcLow, srcLow, mask);
    }
    __asm__ __volatile__("fmax.ps %[resultLow], %[srcLow], %[splatLow]\n"
                         : [ resultLow ] "=f"(resultLow)
                         : FOR_CLANG([ vmask ] "M"(mask)) FOR_CLANG_COMMA[srcLow] "f"(srcLow),
                           [ splatLow ] "f"(splatLow));
    if constexpr (elK != FloatTy) {
      dnn_lib_v2::convert<dnn_lib_v2::v8u32_t, FloatTy, dnn_lib_v2::v8u32_t, elK>(resultLow, resultLow, mask);
    }
  } else {
    assert(isAKnownSuitableForMaxInt64);
    __asm__ __volatile__("flt.pi %[tmp], %[srcLow], %[splatLow]\n"
                         "feq.pi %[tmp2], %[srcHigh], %[splatHigh]\n"
                         "fand.pi %[tmp], %[tmp], %[tmp2]\n"
                         "flt.pi %[tmp2], %[srcHigh], %[splatHigh]\n"
                         "for.pi %[tmp], %[tmp], %[tmp2]\n"
                         "fcmov.ps %[tmp2], %[tmp], %[splatHigh], %[srcHigh]\n"
                         "fcmov.ps %[tmp], %[tmp], %[splatLow], %[srcLow]\n"
                         : [ tmp ] "=&f"(resultLow), [ tmp2 ] "=&f"(resultHigh)
                         : FOR_CLANG([ vmask ] "M"(mask)) FOR_CLANG_COMMA[srcLow] "f"(srcLow),
                           [ splatLow ] "f"(splatLow), [ srcHigh ] "f"(srcHigh), [ splatHigh ] "f"(splatHigh));
  }

  dnn_lib_v2::store<dnn_lib_v2::v8u32_t, bytesPerElement, dstAligned>(dst, srcConf, srcIndices, srcIndicesHigh,
                                                                      resultLow, resultHigh, mask);
}

// Define a dispatcher for the kernel
#define TEMPL_ARGS ElemKind elK, bool srcAligned, bool dstAligned,
#define NAME maxSplatDispatch
#define FUNCTOR maxSplatOp<elK, srcAligned, dstAligned>
DISPATCHER(TEMPL_ARGS, NAME, FUNCTOR)
#undef TEMPL_ARGS
#undef NAME
#undef FUNCTOR

// Generic version is vectorized and threaded
template <ElemKind elK, bool srcAligned, bool dstAligned>
INLINE_ATTR void maxSplatTensor(LibTensor* outT, LibTensor* inT, uint64_t valueBits, uint64_t flags,
                                const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  constexpr size_t length = 8;
  constexpr uint32_t mask = (1 << length) - 1;
#if COMPILER_GCC
  __asm__ __volatile__("mov.m.x m0, zero, %[maskImm]\n" : : [ maskImm ] "i"(mask) :);
#endif

  constexpr size_t bytesPerElement = Type::getElementSize(elK);

  uint64_t srcConf;
  dnn_lib_v2::v8s32_t srcIndices;
  dnn_lib_v2::v8s32_t srcIndicesHigh;
  dnn_lib_v2::setupGatherScatterConfig<bytesPerElement, srcAligned>(srcConf, srcIndices, srcIndicesHigh);

  uint64_t dstConf;
  dnn_lib_v2::v8s32_t dstIndices;
  dnn_lib_v2::v8s32_t dstIndicesHigh;
  dnn_lib_v2::setupGatherScatterConfig<bytesPerElement, dstAligned>(dstConf, dstIndices, dstIndicesHigh);

  dnn_lib_v2::v8u32_t splatLow;
  __asm__ __volatile__("fbcx.ps %[splatLow], %[lower]\n"
                       : [ splatLow ] "=f"(splatLow)
                       : FOR_CLANG([ vmask ] "M"(mask)) FOR_CLANG_COMMA[lower] "r"(valueBits));

  [[maybe_unused]] dnn_lib_v2::v8u32_t splatHigh = {0};
  if constexpr (bytesPerElement > 4) {
    __asm__ __volatile__("fbcx.ps %[splatHigh], %[higher]\n"
                         : [ splatHigh ] "=f"(splatHigh)
                         : FOR_CLANG([ vmask ] "M"(mask)) FOR_CLANG_COMMA[higher] "r"(valueBits >> 32));
  }

  if constexpr (elK == Float16Ty or elK == BFloat16Ty) {
    dnn_lib_v2::convert<dnn_lib_v2::v8u32_t, elK, dnn_lib_v2::v8u32_t, FloatTy>(splatLow, splatLow, mask);
  }

  using elemType = typename elemKind2elemTy<elK>::type;
  maxSplatDispatch<elK, srcAligned, dstAligned>(minionId, activeMinions, flags, outT, elemType{}, std::make_tuple(inT),
                                                std::make_tuple(elemType{}), std::make_index_sequence<1>{}, srcConf,
                                                srcIndices, srcIndicesHigh, dstConf, dstIndices, dstIndicesHigh,
                                                splatLow, splatHigh);
}

// Generic version is vectorized and threaded
template <ElemKind elK>
INLINE_ATTR void fwdLibMaxSplatInst(LibTensor* outT, LibTensor* inT, uint64_t valueBits, uint64_t flags,
                                    const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  constexpr bool srcAligned = false;
  constexpr bool dstAligned = false;
  maxSplatTensor<elK, srcAligned, dstAligned>(outT, inT, valueBits, flags, minionOffset, assignedMinions);
}

template <ElemKind elK>
INLINE_ATTR void fwdLibMaxSplatInstAligned32Bytes(LibTensor* outT, LibTensor* inT, uint64_t valueBits, uint64_t flags,
                                                  const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  constexpr bool srcAligned = true;
  constexpr bool dstAligned = true;
  maxSplatTensor<elK, srcAligned, dstAligned>(outT, inT, valueBits, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // __MAX_SPLAT_INST_H_
