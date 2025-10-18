/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _TRANSPOSE_INST_H_
#define _TRANSPOSE_INST_H_

#include "Float16.h"
#include "LibTensor.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

template <typename srcType, typename std::enable_if<(sizeof(srcType) <= 4), int>::type = 0>
INLINE_ATTR void transposeOp(uintptr_t dst, uintptr_t src, int32_t* scatterValues, int32_t* gatherValues) {
  __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                       ".if %[size] == 4\n"
                       "    fgw.ps  f0, f31(%[src]) \n"
                       ".elseif %[size] == 2\n"
                       "    fgh.ps  f0, f31(%[src]) \n"
                       ".else\n"
                       "    fgb.ps  f0, f31(%[src]) \n"
                       ".endif\n"
                       "flw.ps f31, %[scatterValues] \n"
                       ".if %[size] == 4\n"
                       "    fscw.ps  f0, f31(%[dst]) \n"
                       ".elseif %[size] == 2\n"
                       "    fsch.ps  f0, f31(%[dst]) \n"
                       ".else\n"
                       "    fscb.ps  f0, f31(%[dst]) \n"
                       ".endif\n"
                       :
                       : [ gatherValues ] "m"(*(const int32_t(*)[8])gatherValues),
                         [ scatterValues ] "m"(*(const int32_t(*)[8])scatterValues), [ dst ] "r"(dst), [ src ] "r"(src),
                         [ size ] "i"(sizeof(srcType))
                       : "f0", "f31", "memory");
}

template <typename srcType, typename std::enable_if<(sizeof(srcType) > 4), int>::type = 0>
INLINE_ATTR void transposeOp([[maybe_unused]] uintptr_t dst, [[maybe_unused]] uintptr_t src,
                             [[maybe_unused]] int32_t* scatterValues, [[maybe_unused]] int32_t* gatherValues) {
  //FIXME: TODO: implement
}

  // Vectorized version is the generic
template <ElemKind elK, size_t N>
INLINE_ATTR void fwdLibTransposeInst(LibTensor* outT, LibTensor* inT, const std::array<uint32_t, N>& shuffle,
                                     uint64_t flags, const uint32_t minionOffset = 0,
                                     const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  constexpr size_t typeSize = sizeof(srcType);

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  auto dst = outT->getRawDataPointer<srcType>();
  auto src = inT->getRawDataPointer<srcType>();

  const dim_t *dstIndex = outT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *actPitch = inT->strides().data();

  dim_t srcDimNum = inT->ndims();

  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dst);

  if (maxRead == 0)
    return;

  dim_t newPitch[srcDimNum];
  for (dim_t i = 0; i < srcDimNum; i++)
    newPitch[i] = actPitch[shuffle[i]];

  dim_array_t coord = {0};
  dim_t k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex, k);

  size_t offsetIn = 0;
  size_t offsetOut = 0;
  for (size_t j = 0; j < k; j++) {
    offsetIn += newPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  size_t posMax = maxRead + initialAddr;
  bool done = false;
  dim_t lastDim = srcDimNum - 1;

  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

  size_t newPitchSize = newPitch[lastDim] * typeSize;
  int32_t gatherValues[8];
  for (size_t i = 0; i < 8; i++) {
    gatherValues[i] = static_cast<int32_t>(i * newPitchSize);
  }

  size_t dstPitchSize = dstPitch[lastDim] * typeSize;
  int32_t scatterValues[8];
  for (size_t i = 0; i < 8; i++) {
    scatterValues[i] = static_cast<int32_t>(i * dstPitchSize);
  }

  // Work pending to be done
  while (!done && (offsetOut < posMax)) {
    // Compute number of elements in current row
    size_t elementsInRow = dstIndex[lastDim] - coord[lastDim];
    if ((offsetOut + elementsInRow) > posMax) {
      elementsInRow = posMax - offsetOut;
    }

    // Starting addresses
    uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src) + offsetIn * typeSize;
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dst) + offsetOut * typeSize;

    // Computes full passes and partial passes
    size_t registersInRow = elementsInRow / 8;
    size_t res = elementsInRow - registersInRow * 8;

    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n");
    for (size_t i = 0; i < registersInRow; i++) {
      transposeOp <srcType>(dstAddr, srcAddr, scatterValues, gatherValues);
      srcAddr += 8 * typeSize * newPitch[lastDim];
      dstAddr += 8 * typeSize * dstPitch[lastDim];
    }

    if (res > 0) {
      uint8_t mask = static_cast<uint8_t>((1 << res) - 1);
      __asm__ __volatile__("mov.m.x m0, %[mask], 0\n" : : [mask] "r" (mask) :);
      transposeOp <srcType>(dstAddr, srcAddr, scatterValues, gatherValues);
    }

    // Updates pointers
    if (coord[lastDim] != 0) {
      // Aligning the highest dimension is only required in the first iteration
      // We move offsets to the begining of the second to last dimension
      offsetIn -= coord[lastDim] * newPitch[lastDim];
      offsetOut -= coord[lastDim] * dstPitch[lastDim];
      coord[lastDim] = 0;
    }

    // Increment pointers ignoring the highest dimension as each step takes care of it
    done = getOffsets(lastDim, coord, offsetOut, offsetIn, dstIndex, dstPitch, newPitch);
  }

  // Eviction phase
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr,
                                      clperminion);
}

template <typename srcType, typename std::enable_if<(sizeof(srcType) <= 4), int>::type = 0>
INLINE_ATTR void transposeOpAligned32Bytes(uintptr_t dst, uintptr_t src, int32_t* gatherValues) {
  constexpr size_t size = sizeof(srcType);
  constexpr auto g32_conf = size == 2 ? fg32h_conf : fg32b_conf;

  __asm__ __volatile__("flw.ps f31, %[gatherValues] \n"
                       ".if %[size] == 4\n"
                       "    fgw.ps  f0, f31(%[src]) \n"
                       "    fsw.ps  f0, 0(%[dst]) \n"
                       ".elseif %[size] == 2\n"
                       "    fgh.ps  f0, f31(%[src]) \n"
                       "    li t0, %[g32_conf]\n"
                       "    fsc32h.ps f0, t0(%[dst]) \n"
                       ".else\n"
                       "    fgb.ps  f0, f31(%[src]) \n"
                       "    li t0, %[g32_conf]\n"
                       "    fsc32b.ps  f0, t0(%[dst]) \n"
                       ".endif\n"
                       :
                       : [ gatherValues ] "m"(*(const int32_t(*)[8])gatherValues), [ src ] "r"(src), [ dst ] "r"(dst),
                         [ g32_conf ] "i"(g32_conf), [ size ] "i"(size)
                       : "f0", "f31", "t0", "memory");
}

template <typename srcType, typename std::enable_if<(sizeof(srcType) > 4), int>::type = 0>
INLINE_ATTR void transposeOpAligned32Bytes([[maybe_unused]] uintptr_t dst, [[maybe_unused]] uintptr_t src,
                                           [[maybe_unused]] int32_t* gatherValues) {
  //FIXME: not implemented
}

template <ElemKind elK, size_t N>
INLINE_ATTR void
fwdLibTransposeInstAligned32Bytes(LibTensor* outT, LibTensor* inT, const std::array<uint32_t, N>& shuffle,
                                  uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer();
  void* src = inT->getRawDataPointer();

  uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dst);
  uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src);

  const dim_t *dstIndex = outT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *actPitch = inT->strides().data();

  dim_t srcDimNum = inT->ndims();

  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = sizeof(srcType);
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dst);
  if (maxRead == 0) {
    return;
  }

  dim_t newPitch[srcDimNum];
  dim_t newdstPitch[srcDimNum];
  dim_t newdstIndex[srcDimNum];
  for (dim_t i = 0; i < srcDimNum; i++) {
    newPitch[i] = actPitch[shuffle[i]];
    newdstPitch[i] = dstPitch[i];
    newdstIndex[i] = dstIndex[i];
  }

  dim_array_t coord = {0};
  dim_t k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex, k);

  size_t offsetIn = 0;
  size_t offsetOut = 0;
  for (dim_t j = 0; j < k; j++) {
    offsetIn += newPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  size_t posMax = maxRead + initialAddr;
  dim_t lastDim = srcDimNum - 1;
  size_t newPitchSize = newPitch[lastDim] * typeSize;
  int32_t gatherValues[8];
  for (dim_t i = 0; i < 8; i++) {
    gatherValues[i] = static_cast<int32_t>(i * newPitchSize);
  }

  // We modify the pitches and coord so that the function getOffsets
  // jumps eight positions in lastDim, the smallest dimension.
  // Number 8 is the amount of lanes that a register has.
  size_t res = ((dstIndex[lastDim] - 1) % 8) + 1;
  coord[lastDim] /= 8;
  newPitch[lastDim] *= 8;
  newdstPitch[lastDim] *= 8;
  newdstIndex[lastDim] = ((dstIndex[lastDim] - 1) / 8) + 1;
  uint8_t mask = static_cast<uint8_t>((1 << res) - 1);

  bool done = false;
  while (!done && (offsetOut < posMax)) {
    dstAddr = reinterpret_cast<uintptr_t>(dst) + offsetOut*typeSize;
    srcAddr = reinterpret_cast<uintptr_t>(src) + offsetIn*typeSize;

    // When the minion reaches the end of the lastDim, we use a mask
    // that is always the same because the dst Tensor is aligned to 32 Bytes.
    if (coord[lastDim] != newdstIndex[lastDim] - 1) {
      __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
    } else {
      __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
    }

    transposeOpAligned32Bytes <srcType>(dstAddr, srcAddr, gatherValues);
    done = getOffsets(srcDimNum, coord, offsetOut, offsetIn, newdstIndex, newdstPitch, newPitch);
  }
  if (DO_EVICTS) {
    size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
    if (clperminion > 0) {
      fence_evict_va(0, DO_EVICTS, initialAddr, clperminion - 1, CACHE_LINE_BYTES);
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif //  _TRANSPOSE_INST_H_
