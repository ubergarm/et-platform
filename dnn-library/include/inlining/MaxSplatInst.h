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
inline void maxSplatOp(const uintptr_t dst, const uintptr_t src, const dim_t valid, const float splatVal,
                       const float* scale, const int32_t* offset) {
  // Enables only the valid elements
  if (valid < 8) {
    uint8_t mask = ((1 << valid) - 1);
    __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
  } else {
    __asm__ __volatile__("mov.m.x m0, zero, 0xFF \n");
  }

  constexpr size_t bytesPerElement = Type::getElementSize(elK);

  // TODO: move splatVal to template parameter known at build time
  uint64_t splatValueScalar = bitwise_copy<uint32_t>(splatVal);

  float splatValueVector;
  __asm__ __volatile__("fbcx.ps %[splatValueVector], %[splatValueScalar]\n"
                       : [ splatValueVector ] "=f"(splatValueVector)
                       : [ splatValueScalar ] "r"(splatValueScalar));

  uint64_t conf;
  float indices;
  float indicesHigh;
  setupGatherScatterConfig<bytesPerElement, aligned>(conf, indices, indicesHigh);

  float op0 = 0.f, op0High = 0;

  load<bytesPerElement, aligned>(src, conf, indices, indicesHigh, op0, op0High);

  float scale_v, offset_v;
  (void)scale_v;
  (void)offset_v;

  if constexpr (isQuantizedElemKind(elK)) {
    __asm__ __volatile__("fbcx.ps %[offset], %[offset_s] \n"
                         "fbcx.ps %[scale], %[scale_s] \n"
                         "fsub.pi %[op0], %[op0], %[offset] \n"
                         "fcvt.ps.pw %[op0], %[op0] \n"
                         "fmul.ps %[op0], %[op0], %[scale] \n"
                         : [ op0 ] "+&f"(op0), [ scale ] "=&f"(scale_v), [ offset ] "=&f"(offset_v)
                         : [ scale_s ] "r"(bitwise_copy<uint32_t>(scale[0])), [ offset_s ] "r"(offset[0]));
  } else if constexpr (elK == Float16Ty) {
    __asm__ __volatile__("fcvt.ps.f16 %[op0], %[op0]\n" : [ op0 ] "+f"(op0) :);
  } else if constexpr (elK == BFloat16Ty) {
    __asm__ __volatile__("fslli.pi %[op0], %[op0], %[bits]\n" : [ op0 ] "+f"(op0) : [ bits ] "i"(16));
  }

  __asm__ __volatile__("fmax.ps %[op0], %[op0], %[splatValueVector]\n"
                       : [ op0 ] "+f"(op0)
                       : [ splatValueVector ] "f"(splatValueVector));

  if constexpr (isQuantizedElemKind(elK)) {
    __asm__ __volatile__("frcp.ps %[scale], %[scale]\n"
                         "fcvt.ps.pw %[offset], %[offset]\n"
                         "fmadd.ps %[op0], %[op0], %[scale], %[offset]\n"
                         "fcvt.pw.ps %[op0], %[op0]\n"
                         : [ op0 ] "+f"(op0), [ scale ] "+f"(scale_v), [ offset ] "+f"(offset_v)
                         :);
    if constexpr (elK == Int8QTy) {
      __asm__ __volatile__("fsat8.pi %[op0], %[op0]\n" : [ op0 ] "+f"(op0) :);
    }
  } else if constexpr (elK == Float16Ty) {
    __asm__ __volatile__("fcvt.f16.ps %[op0], %[op0]\n" : [ op0 ] "+f"(op0) :);
  } else if constexpr (elK == BFloat16Ty) {
    __asm__ __volatile__("fsrli.pi %[op0], %[op0], %[bits]\n" : [ op0 ] "+f"(op0) : [ bits ] "i"(16));
  }

  store<bytesPerElement, aligned>(dst, conf, indices, indicesHigh, op0, op0High);
}

// Generic version is vectorized and threaded
template <ElemKind elK>
inline void fwdLibMaxSplatInst(LibTensor* outT, LibTensor* inT, const float splatVal, uint64_t flags,
                               const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  constexpr bool aligned = false;

  const float scale[2] = {inT->getScale(), outT->getScale()};
  const int32_t offset[2] = {inT->getOffset(), outT->getOffset()};

  outT->partitionLoop<srcType>(minionId, activeMinions, flags, inT, maxSplatOp<elK, aligned>, splatVal, scale, offset);
}

template <ElemKind elK>
inline void fwdLibMaxSplatInstAligned32Bytes(LibTensor* outT, LibTensor* inT, const float splatVal, uint64_t flags,
                                             const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  constexpr bool aligned = true;

  const float scale[2] = {inT->getScale(), outT->getScale()};
  const int32_t offset[2] = {inT->getOffset(), outT->getOffset()};

  outT->partitionLoop<srcType>(minionId, activeMinions, flags, inT, maxSplatOp<elK, aligned>, splatVal, scale, offset);
}

} // namespace inlining

} // namespace dnn_lib

#endif // __MAX_SPLAT_INST_H_
