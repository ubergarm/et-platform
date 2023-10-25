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

#ifndef _MAX_POOL_INST_H_
#define _MAX_POOL_INST_H_

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

template <typename T> INLINE_ATTR bool greaterOrEqualThan(T a, T b) {
  return a >= b;
}

using StorageForFloat16Ty = elemKind2elemTy<Float16Ty>::type;

INLINE_ATTR bool greaterOrEqualThanFloat16(StorageForFloat16Ty a, StorageForFloat16Ty b) {
  float af, bf;
  dnn_lib::convertFp16ToFp32(a, af);
  dnn_lib::convertFp16ToFp32(b, bf);
  return af >= bf;
}

using StorageForBFloat16Ty = elemKind2elemTy<BFloat16Ty>::type;

INLINE_ATTR bool greaterOrEqualThanBFloat16(StorageForBFloat16Ty a, StorageForBFloat16Ty b) {
  float af, bf;
  dnn_lib::convertBfloat16ToFp32(a, af);
  dnn_lib::convertBfloat16ToFp32(b, bf);
  return af >= bf;
}

template <ElemKind elK, typename... Types> INLINE_ATTR bool greaterOrEqualThan(Types... args) {
  if constexpr (elK == Float16Ty) {
    return greaterOrEqualThanFloat16(args...);
  } else if constexpr (elK == BFloat16Ty) {
    return greaterOrEqualThanBFloat16(args...);
  } else {
    return greaterOrEqualThan(args...);
  }
  assert(false);
  return false;
}

template <ElemKind dstElK, ElemKind srcElK, size_t N, size_t PN>
INLINE_ATTR void maxPoolImplThreaded(bool argMax, LibTensor* outT, LibTensor* out2T, LibTensor* inT,
                                     const std::array<uint32_t, N>& kernels, const std::array<uint32_t, N>& strides,
                                     const std::array<uint32_t, PN>& pads, uint64_t flags,
                                     const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  using dstType = typename elemKind2elemTy<dstElK>::type;
  using srcType = typename elemKind2elemTy<srcElK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  /* maintain compatibility through the new Iface Libtensor */

  dstType* tOutput = outT->getRawDataPointer<dstType>();
  srcType* tInput = inT->getRawDataPointer<srcType>();
  int64_t* tOutput2 = (out2T != nullptr) ? out2T->getRawDataPointer<int64_t>() : nullptr;

  const dim_t *dstIndex = outT->dims().data();
  const dim_t *actIndex = inT->dims().data();
  
  const dim_t *dstPitch = outT->strides().data();
  const dim_t* dst2Pitch = (out2T != nullptr) ? out2T->strides().data() : nullptr;
  const dim_t *actPitch = inT->strides().data(); 

  auto srcPitchNoPadding = inT->stridesNoPadding();

  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, tOutput);
  if (maxRead == 0)
    return;

  dim_array_t coord = {0};
  dim_t k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 4, dstPitch, dstIndex, k);

  size_t offsetOut = 0;
  for (dim_t i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  size_t posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y;

  srcType max_value;

  // Vector instructions working as scalar
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n");

  // Setup scatter configuration for destination
  constexpr size_t dstBytesPerElement = Type::getElementSize(dstElK);
  uint64_t dstConf;
  float dstIndices, dstIndicesHigh;
  setupGatherScatterConfig<dstBytesPerElement, false>(dstConf, dstIndices, dstIndicesHigh);

  // Setup gather configuration for source
  constexpr size_t srcBytesPerElement = Type::getElementSize(srcElK);
  uint64_t srcConf;
  float srcIndices, srcIndicesHigh;
  setupGatherScatterConfig<srcBytesPerElement, false>(srcConf, srcIndices, srcIndicesHigh);

  // Setup source dequantize
  float srcScale, srcOffset;
  setupDequantize(srcScale, srcOffset, inT->getScale(), inT->getOffset());

  // Setup destination quantize
  float dstScaleReciprocal, dstOffset;
  setupQuantize(dstScaleReciprocal, dstOffset, outT->getScale(), outT->getOffset());

  while (!done && (offsetOut < posMax)) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);

    bool first = true;
    // When the MaxPool window includes only padding pixels then for that
    // window by convention we return 0  /(offset for quantized types).
    max_value = outT->getType().isQuantizedType() ? static_cast<srcType>(outT->getOffset()) : 0;
    int64_t argmaxNHWC = 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {
      for (size_t fy = 0; fy < kernels[1]; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }

        size_t idx =
          coord[0] * actPitch[0] + (size_t)ox * actPitch[1] + (size_t)oy * actPitch[2] + coord[3] * actPitch[3];
        srcType val = tInput[idx];
        if (first || greaterOrEqualThan<srcElK>(val, max_value)) {
          first = false;
          max_value = val;
          if (argMax) 
            argmaxNHWC = coord[0] * srcPitchNoPadding[0] +
              (size_t)ox * srcPitchNoPadding[1] +
              (size_t)oy * srcPitchNoPadding[2]
              + coord[3];
        }
      }
    }

    // Load result in srcElK format, convert to dstElK and store
    uintptr_t srcAddr = reinterpret_cast<uintptr_t>(&max_value);
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(&tOutput[offsetOut]);
    float value, valueHigh;
    load<srcBytesPerElement, false>(srcAddr, srcConf, srcIndices, srcIndicesHigh, value, valueHigh);
    convert<srcElK, dstElK>(value, valueHigh, value, valueHigh, srcScale, srcOffset, dstScaleReciprocal, dstOffset);
    store<dstBytesPerElement, false>(dstAddr, dstConf, dstIndices, dstIndicesHigh, value, valueHigh);

    if (argMax) {
      int64_t dst2Addr = coord[0] * dst2Pitch[0] + coord[1] * dst2Pitch[1] +
                         coord[2] * dst2Pitch[2] + coord[3] * dst2Pitch[3];
      tOutput2[dst2Addr] = argmaxNHWC;
    }
    done = getOffsets(4, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0)
    evict_va_multi(DO_EVICTS, (uintptr_t)tOutput + typeSize * initialAddr, clperminion);
}

  ////////////////////////////////////////////////////////////////////////////////
  // Max Pool instruction
  ////////////////////////////////////////////////////////////////////////////////

template <ElemKind out0Type, ElemKind in0Type, size_t N, size_t PN>
INLINE_ATTR void fwdLibMaxPoolInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, N>& kernels,
                                   const std::array<uint32_t, N>& strides, const std::array<uint32_t, PN>& pads,
                                   [[maybe_unused]] uint32_t layout, uint64_t flags, const uint32_t minionOffset = 0,
                                   const uint32_t assignedMinions = 0) {
  maxPoolImplThreaded<out0Type, in0Type>(false, out0, nullptr, in0, kernels, strides, pads, flags, minionOffset,
                                         assignedMinions);
}

  ////////////////////////////////////////////////////////////////////////////////
  // Max Pool with ARGMAX instruction
  ////////////////////////////////////////////////////////////////////////////////

template <ElemKind out0Type, ElemKind in0Type, size_t N, size_t PN>
INLINE_ATTR void
fwdLibMaxPoolWithArgMaxInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, N>& kernels,
                            const std::array<uint32_t, N>& strides, const std::array<uint32_t, PN>& pads,
                            [[maybe_unused]] uint32_t layout, uint64_t flags, const uint32_t minionOffset = 0,
                            const uint32_t assignedMinions = 0) {
  maxPoolImplThreaded<out0Type, in0Type>(true, out0, out1, in0, kernels, strides, pads, flags, minionOffset,
                                         assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _MAX_POOL_INST_H_
