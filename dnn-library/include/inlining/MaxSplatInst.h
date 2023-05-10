/*-------------------------------------------------------------------------
 * Copyright (C) 2020, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef __MAX_SPLAT_INST_H_
#define __MAX_SPLAT_INST_H_

#include "Float16.h"
#include "LibTensor.h"
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
INLINE_ATTR void maxSplatOp(const uintptr_t dst, [[maybe_unused]] const uintptr_t src, const dim_t valid,
                            [[maybe_unused]] uint64_t srcConf, [[maybe_unused]] float srcIndices,
                            [[maybe_unused]] float srcIndicesHigh, [[maybe_unused]] uint64_t dstConf,
                            [[maybe_unused]] float dstIndices, [[maybe_unused]] float dstIndicesHigh, float splatLow,
                            [[maybe_unused]] float splatHigh) {

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
  __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"((1 << valid) - 1));

  float srcLow, srcHigh;
  load<bytesPerElement, srcAligned>(src, srcConf, srcIndices, srcIndicesHigh, srcLow, srcHigh);

  float resultLow;
  float resultHigh = 0;
  if constexpr (isAKnownSuitableForMaxInt32) {
    __asm__ __volatile__("fmax.pi %[resultLow], %[srcLow], %[splatLow]\n"
                         : [ resultLow ] "=f"(resultLow)
                         : [ srcLow ] "f"(srcLow), [ splatLow ] "f"(splatLow));
  } else if constexpr (isAKnownSuitableForMaxUInt32) {
    __asm__ __volatile__("fmaxu.pi %[resultLow], %[srcLow], %[splatLow]\n"
                         : [ resultLow ] "=f"(resultLow)
                         : [ srcLow ] "f"(srcLow), [ splatLow ] "f"(splatLow));
  } else if constexpr (isAKnownSuitableForMaxFloat) {
    if constexpr (elK != FloatTy) {
      convert<elK, FloatTy>(srcLow, srcLow);
    }
    __asm__ __volatile__("fmax.ps %[resultLow], %[srcLow], %[splatLow]\n"
                         : [ resultLow ] "=f"(resultLow)
                         : [ srcLow ] "f"(srcLow), [ splatLow ] "f"(splatLow));
    if constexpr (elK != FloatTy) {
      convert<FloatTy, elK>(resultLow, resultLow);
    }
  } else if constexpr (isAKnownSuitableForMaxInt64) {
    __asm__ __volatile__("flt.pi %[tmp], %[srcLow], %[splatLow]\n"
                         "feq.pi %[tmp2], %[srcHigh], %[splatHigh]\n"
                         "fand.pi %[tmp], %[tmp], %[tmp2]\n"
                         "flt.pi %[tmp2], %[srcHigh], %[splatHigh]\n"
                         "for.pi %[tmp], %[tmp], %[tmp2]\n"
                         "fcmov.ps %[tmp2], %[tmp], %[splatHigh], %[srcHigh]\n"
                         "fcmov.ps %[tmp], %[tmp], %[splatLow], %[srcLow]\n"
                         : [ tmp ] "=&f"(resultLow), [ tmp2 ] "=&f"(resultHigh)
                         : [ srcLow ] "f"(srcLow), [ splatLow ] "f"(splatLow), [ srcHigh ] "f"(srcHigh),
                           [ splatHigh ] "f"(splatHigh));
  }

  store<bytesPerElement, dstAligned>(dst, srcConf, srcIndices, srcIndicesHigh, resultLow, resultHigh);
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

  constexpr size_t bytesPerElement = Type::getElementSize(elK);

  __asm__ __volatile__("mov.m.x m0, zero, 0xff\n");

  uint64_t srcConf;
  float srcIndices;
  float srcIndicesHigh;
  setupGatherScatterConfig<bytesPerElement, srcAligned>(srcConf, srcIndices, srcIndicesHigh);

  uint64_t dstConf;
  float dstIndices;
  float dstIndicesHigh;
  setupGatherScatterConfig<bytesPerElement, dstAligned>(dstConf, dstIndices, dstIndicesHigh);

  float splatLow;
  __asm__ __volatile__("fbcx.ps %[splatLow], %[lower]\n" : [ splatLow ] "=f"(splatLow) : [ lower ] "r"(valueBits));

  [[maybe_unused]] float splatHigh = 0.f;
  if constexpr (bytesPerElement > 4) {
    __asm__ __volatile__("fbcx.ps %[splatHigh], %[higher]\n"
                         : [ splatHigh ] "=f"(splatHigh)
                         : [ higher ] "r"(valueBits >> 32));
  }

  if constexpr (elK == Float16Ty or elK == BFloat16Ty) {
    convert<elK, FloatTy>(splatLow, splatLow);
  }

#if 0
  outT->partitionLoop<srcType>(minionId, activeMinions, flags, inT, maxSplatOp<elK, srcAligned, dstAligned>,   
                       srcConf, srcIndices, srcIndicesHigh,
                       dstConf, dstIndices, dstIndicesHigh,
                       splatLow, splatHigh);
#else
  using elemType = typename elemKind2elemTy<elK>::type;
  maxSplatDispatch<elK, srcAligned, dstAligned>(minionId, activeMinions, flags, outT, elemType{}, std::make_tuple(inT),
                                                std::make_tuple(elemType{}), std::make_index_sequence<1>{}, srcConf,
                                                srcIndices, srcIndicesHigh, dstConf, dstIndices, dstIndicesHigh,
                                                splatLow, splatHigh);
#endif
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
