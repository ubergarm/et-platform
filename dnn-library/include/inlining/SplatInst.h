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

template <ElemKind elK, bool aligned>
inline void splatOp(const uintptr_t dst, const uintptr_t src, const dim_t valid, const float splatVal,
                    const float scale, const int32_t offset) {
  // Enable only the valid elements
  if (valid < 8) {
    uint8_t mask = ((1 << valid) - 1);
    __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
  } else {
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
  }

  constexpr size_t bytesPerElement = Type::getElementSize(elK);

  // TODO: move splatVal to template parameter known at build time
  uint64_t splatValueScalar = bitwise_copy<uint32_t>(splatVal);

  float splatValueVector;
  [[maybe_unused]] float splatValueVectorHigh = 0.f;
  __asm__ __volatile__("fbcx.ps %[splatValueVector], %[splatValueScalar]\n"
                       : [ splatValueVector ] "=f"(splatValueVector)
                       : [ splatValueScalar ] "r"(splatValueScalar));

  [[maybe_unused]] float srcScale;
  [[maybe_unused]] float srcOffset;

  [[maybe_unused]] float dstScaleReciprocal;
  [[maybe_unused]] float dstOffset;
  if constexpr (isQuantizedElemKind(elK)) {
    setupQuantize(dstScaleReciprocal, dstOffset, scale, offset);
  }

  uint64_t conf;
  float indices;
  float indicesHigh;
  setupGatherScatterConfig<bytesPerElement, aligned>(conf, indices, indicesHigh);

  float op = 0.f;
  [[maybe_unused]] float opHigh = 0.f;
  convert<FloatTy, elK>(splatValueVector, splatValueVectorHigh, op, opHigh, srcScale, srcOffset, dstScaleReciprocal,
                        dstOffset);

  store<bytesPerElement, aligned>(dst, conf, indices, indicesHigh, op, opHigh);
}

template <ElemKind elK>
inline void fwdLibSplatInst(LibTensor* outT, const float splatVal, uint64_t flags, const uint32_t minionOffset = 0,
                            const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  constexpr bool aligned = false;

  const float scale = outT->getScale();
  const int32_t offset = outT->getOffset();

  outT->partitionLoop<srcType>(minionId, activeMinions, flags, outT, splatOp<elK, aligned>, splatVal, scale, offset);
}

template <ElemKind elK>
inline void fwdLibSplatInstAligned32Bytes(LibTensor* outT, const float splatVal, uint64_t flags,
                                          const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  constexpr bool aligned = true;

  const float scale = outT->getScale();
  const int32_t offset = outT->getOffset();

  outT->partitionLoop<srcType>(minionId, activeMinions, flags, outT, splatOp<elK, aligned>, splatVal, scale, offset);
}

} // namespace inlining

} // namespace dnn_lib

#endif // __SPLAT_INST_H_
