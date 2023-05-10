/*-------------------------------------------------------------------------
 * Copyright (C) 2021, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef __SPLAT_INST_H_
#define __SPLAT_INST_H_

#include "LibTensor.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

template <ElemKind elK, bool aligned>
INLINE_ATTR void splatOp(const uintptr_t dst, [[maybe_unused]] const uintptr_t src, const dim_t valid,
                         [[maybe_unused]] uint64_t conf, [[maybe_unused]] float indices,
                         [[maybe_unused]] float indicesHigh, float op, [[maybe_unused]] float opHigh) {

  constexpr size_t bytesPerElement = Type::getElementSize(elK);

  // Enable only the valid elements
  __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"((1 << valid) - 1) :);

  store<bytesPerElement, aligned>(dst, conf, indices, indicesHigh, op, opHigh);
}

template <ElemKind elK, bool aligned>
INLINE_ATTR void splatTensor(LibTensor* outT, const uint64_t splatVal, uint64_t flags, const uint32_t minionOffset = 0,
                             const uint32_t assignedMinions = 0) {

  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  constexpr size_t bytesPerElement = Type::getElementSize(elK);

  __asm__ __volatile__("mov.m.x m0, zero, 0xff\n");

  uint64_t conf;
  float indices;
  float indicesHigh;
  setupGatherScatterConfig<bytesPerElement, aligned>(conf, indices, indicesHigh);

  float op = 0.f;
  __asm__ __volatile__("fbcx.ps %[op], %[lower]\n" : [ op ] "=f"(op) : [ lower ] "r"(splatVal));

  [[maybe_unused]] float opHigh = 0.f;
  if constexpr (bytesPerElement > 4) {
    __asm__ __volatile__("fbcx.ps %[opHigh], %[higher]\n" : [ opHigh ] "=f"(opHigh) : [ higher ] "r"(splatVal >> 32));
  }

  outT->partitionLoop<srcType>(minionId, activeMinions, flags, outT, splatOp<elK, aligned>, conf, indices, indicesHigh,
                               op, opHigh);
}

template <ElemKind elK>
INLINE_ATTR void fwdLibSplatInst(LibTensor* outT, const uint64_t splatVal, uint64_t flags,
                                 const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  constexpr bool aligned = false;
  splatTensor<elK, aligned>(outT, splatVal, flags, minionOffset, assignedMinions);
}

template <ElemKind elK>
INLINE_ATTR void fwdLibSplatInstAligned32Bytes(LibTensor* outT, const uint64_t splatVal, uint64_t flags,
                                               const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  constexpr bool aligned = true;
  splatTensor<elK, aligned>(outT, splatVal, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // __SPLAT_INST_H_
