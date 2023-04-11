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

#ifndef _SPARSE_TO_DENSE_INST_H_
#define _SPARSE_TO_DENSE_INST_H_

#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

#include "Float16.h"
#include "utils.h" // From include/internal path
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
inline __attribute__((always_inline)) void sparseToDenseOp(uintptr_t dst, uintptr_t src, const uint64_t* indices,
                                                           size_t batchPitchBytes, dim_t batchElement,
                                                           dim_t numIndices) {

  constexpr size_t bytesPerElement = Type::getElementSize(elK);

  uint64_t conf;
  float offset;
  float offsetHigh;
  setupGatherScatterConfig<bytesPerElement, false>(conf, offset, offsetHigh);

  const uint64_t* indexPtr = indices;
  const uint64_t* indexPtrEnd = indices + numIndices;

  uintptr_t srcAddress = src;

  // Reset accumulator
  float accum, accumHigh;
  zero<elK>(accum, accumHigh);

  do {
    // If current index matchs the batch element load and accumulate
    if (*indexPtr == batchElement) {

      // Load value
      float value, valueHigh;
      load<bytesPerElement, false>(srcAddress, conf, offset, offsetHigh, value, valueHigh);

      // Convert to intermediate format (float or int32)
      if constexpr (elK == Float16Ty) {
        __asm__ __volatile__("fcvt.ps.f16 %[value], %[value]\n" : [ value ] "+f"(value));
      } else if constexpr (elK == BFloat16Ty) {
        __asm__ __volatile__("fslli.pi %[value], %[value], 16\n" : [ value ] "+f"(value));
      } else if constexpr (elK == UInt8QTy) {
        __asm__ __volatile__("fsatu8.pi %[value], %[value]\n" : [ value ] "+f"(value));
      }

      // Accumulate differently depending on the intermediate value
      if constexpr (elK == FloatTy or elK == Float16Ty or elK == BFloat16Ty) {
        // Intermediate is float
        __asm__ __volatile__("fadd.ps %[accum], %[accum], %[value]\n" : [ accum ] "+f"(accum) : [ value ] "f"(value));
      } else if constexpr (elK == Int64ITy) {
        // Intermediate is int64_t
        float carry;
        __asm__ __volatile__(
          // Determine whether there is carry from lower to higher 32 bits
          "fnot.pi %[carry], %[accum]\n"
          "fltu.pi %[carry], %[carry], %[value]\n"
          // Add lower 32 bits
          "fadd.pi %[accum], %[accum], %[value]\n"
          // Add high 32 bits
          "fsub.pi %[accumHigh], %[accumHigh], %[carry]\n"
          "fadd.pi %[accumHigh], %[accumHigh], %[valueHigh]\n"
          : [ carry ] "=&f"(carry), [ accum ] "+f"(accum), [ accumHigh ] "+f"(accumHigh), [ value ] "+f"(value),
            [ valueHigh ] "+f"(valueHigh));
      } else {
        // Intermediate is int32_t
        __asm__ __volatile__("fadd.pi %[accum], %[accum], %[value]\n" : [ accum ] "+f"(accum) : [ value ] "f"(value));
      }
    }

    srcAddress += batchPitchBytes;
    indexPtr++;

  } while (indexPtr != indexPtrEnd);

  // Convert from intermediate format (float, int32 or int64) to elK
  if constexpr (elK == Float16Ty) {
    __asm__ __volatile__("fcvt.f16.ps %[accum], %[accum]\n" : [ accum ] "+f"(accum));
  } else if constexpr (elK == BFloat16Ty) {
    __asm__ __volatile__("fsrli.pi %[accum], %[accum], 16\n" : [ accum ] "+f"(accum));
  } else if constexpr (elK == Int8QTy) {
    __asm__ __volatile__("fsat8.pi %[accum], %[accum]\n" : [ accum ] "+f"(accum));
  }

  // Store
  store<bytesPerElement>(dst, conf, offset, offsetHigh, accum, accumHigh);
}

// Vectorized version
template <ElemKind elK>
inline __attribute__((always_inline)) void fwdLibSparseToDenseInst(
                                                                   LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                                                   uint64_t flags,
                                                                   const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* outT --> dst  in2T--> src   in1T--> indices */
  /* maintain compatibility through the new Iface Libtensor */

  void* dstT = outT->getRawDataPointer();
  void* srcT = in2T->getRawDataPointer();
  const uint64_t* indices = in1T->getRawDataPointer<uint64_t>();

  const dim_t *dstIndex = outT->dims().data();
  const dim_t *indIndex = in1T->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *srcPitch = in2T->strides().data();

  dim_t srcDimNum = in2T->ndims();

  uintptr_t dstAddr = (uintptr_t)dstT;    
  uintptr_t srcAddr = (uintptr_t)srcT;

  size_t numElemsDst = dstPitch[0] * dstIndex[0];

  size_t initialAddr, maxRead;
  constexpr size_t typeSize = Type::getElementSize(elK);
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  dim_array_t coord = {0};
  for (dim_t i = 0; i < srcDimNum; i++)
    coord[i] = 0;
  dim_t k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex, k);

  size_t offsetIn = 0;
  size_t offsetOut = 0;

  size_t batchPitchBytes = srcPitch[0] * Type::getElementSize(elK);
  // @TODO srcpitch It is a cnst pointer!!!!. Re-do in other way
  // it is not allowed modify tensor properties. It needs a cpy of it.
  size_t cpySrcPitch[srcDimNum];
  for (size_t i = 0; i < srcDimNum; i++)
    cpySrcPitch[i] = srcPitch[i];
  //srcPitch[0] = 0;
  cpySrcPitch[0] = 0;

  for (dim_t j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
    offsetIn += cpySrcPitch[j] * coord[j];
  }

  size_t posMax = maxRead + initialAddr;
  bool done = false;
  size_t lastDim = srcDimNum - 1;
  size_t maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  size_t elementsInRow, registersInRow, res;

  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  coord[0] *= (srcDimNum != 1);

  while (!done && (offsetOut < posMax)) {
    if (firstRow && (srcDimNum > 1) && coord[lastDim - 1] != maxRow) {
      elementsInRow = dstIndex[lastDim] - coord[lastDim];
    } else if ((srcDimNum == 1) || (coord[lastDim - 1] == maxRow)) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
      if (elementsInRow > (dstIndex[lastDim] - coord[lastDim])) {
        elementsInRow = dstIndex[lastDim] - coord[lastDim];
      }
    } else {
      elementsInRow = dstIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      res = elementsInRow - registersInRow * 8;
      uint8_t mask = static_cast<uint8_t>((1 << res) - 1);
      __asm__ __volatile__("mov.m.x m1, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr += offsetIn * typeSize;
    dstAddr += offsetOut * typeSize;

    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

    for (size_t i = 0; i < registersInRow; i++) {
      sparseToDenseOp<elK>(dstAddr, srcAddr, indices, batchPitchBytes, coord[0], indIndex[0]);
      srcAddr += 8 * typeSize;
      dstAddr += 8 * typeSize;
    }
    if(res > 0) {
      __asm__ __volatile__("maskand m0, m1, m0 \n");
      sparseToDenseOp<elK>(dstAddr, srcAddr, indices, batchPitchBytes, coord[0], indIndex[0]);
    }
    if (lastRow)
      return;

    dstAddr = (uintptr_t)dstT;
    srcAddr = (uintptr_t)srcT;

    offsetIn -= coord[lastDim] * cpySrcPitch[lastDim];    
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;

    done = getOffsets(lastDim , coord, offsetIn, offsetOut, dstIndex,
                      cpySrcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;

  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _SPARSE_TO_DENSE_INST_H_
