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

#ifndef _INSERT_TENSOR_INST_H_
#define _INSERT_TENSOR_INST_H_

#include "Addresser.h"
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

template <ElemKind elK>
INLINE_ATTR void fwdLibInsertTensorInst(LibTensor* outT, LibTensor* inT, const dim_array_t offsets, uint32_t count,
                                        dim_t axis, uint64_t flags, const uint32_t minionOffset = 0,
                                        [[maybe_unused]] const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  if (get_minion_id() != minionOffset) return;
  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer();
  void* src = inT->getRawDataPointer();

  // Addresser<elK> tOutput(dst, scale[1], offset[1]);
  Addresser<elK> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tSmallInput(src2, scale[0], offset[0]);
  const Addresser<elK> tSmallInput(src, inT->getScale(), inT->getOffset());
  
  const dim_array_t &eDims = inT->dims();
  const dim_array_t &eDstPitch = outT->strides();
  const dim_array_t &eSrcPitch = inT->strides();
  
  size_t advanceOnAxis = 0;
  uint64_t idx;
  uintptr_t addr2wr = 0, previous_addr2wr = (uintptr_t)dst;

  for (size_t cnt = 0; cnt < count; cnt++) {
    // We can use this loop for all shapes.
    for (size_t x = 0; x < eDims[0]; x++) {
      for (size_t y = 0; y < eDims[1]; y++) {
        for (size_t z = 0; z < eDims[2]; z++) {
          for (size_t w = 0; w < eDims[3]; w++) {
            for (size_t q = 0; q < eDims[4]; q++) {
              for (size_t r = 0; r < eDims[5]; r++) {

                idx = (offsets[0] + x) * eDstPitch[0] +
                  (offsets[1] + y) * eDstPitch[1] +
                  (offsets[2] + z) * eDstPitch[2] +
                  (offsets[3] + w) * eDstPitch[3] +
                  (offsets[4] + q) * eDstPitch[4] +
                  (offsets[5] + r) * eDstPitch[5] + advanceOnAxis;

                tOutput[idx] = tSmallInput[x * eSrcPitch[0] + y * eSrcPitch[1] +
                                           z * eSrcPitch[2] + w * eSrcPitch[3] +
                                           q * eSrcPitch[4] + r * eSrcPitch[5]];
                if (DO_EVICTS) {
                  addr2wr = ((uintptr_t)dst) + idx*getsize<srcType>();
                  if ((addr2wr >> 6) != (previous_addr2wr >> 6))  
                  {
                    /* evict current cache line */
                    if (previous_addr2wr != 0) fence_evict_va(0, DO_EVICTS, previous_addr2wr, 0);
                    previous_addr2wr = addr2wr;
                  }
                }
              }
            }
          }
        }
      }
    }
    advanceOnAxis += eDstPitch[axis] * eDims[axis];
  }

  if (DO_EVICTS) {
    if (addr2wr > 0)
      fence_evict_va(0, DO_EVICTS, addr2wr, 0);
  }
}

template <typename srcType>
INLINE_ATTR void insertRow(uint8_t* dst, uint8_t* src, const size_t& addrOut, const size_t& addrIn,
                           const int32_t& typeSize, std::pair<size_t, size_t> lanes, int32_t* gatherValues,
                           uint64_t flags) {
  uint8_t *dst8 = (uint8_t *) dst + addrOut * typeSize;
  uint8_t *src8 = (uint8_t *) src + addrIn * typeSize;
  float scratch;

  uintptr_t addr2evict = (uintptr_t)dst8;
  // Computes bytes to evict (adds the unaligned cache line bytes of the base address
  size_t bytes2evict = ((lanes.first * 4UL) + (addr2evict & 0x3F));
  // With that computes the cl2evict
  size_t cl2evict = (bytes2evict + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;

  while (lanes.first > 8) {

    __asm__ __volatile__("flq2 %[d], %[src]\n"
                         "fsq2 %[d], %[dst]\n"
                         : [ dst ] "=m"(*(uint8_t(*)[32]) dst8), [d] "=&f" (scratch)
                         : [ src ] "m" ( *(const uint8_t(*)[32]) src8)
                         );
    lanes.first -= 8;
    src8 += 32;
    dst8 += 32;
  }
  __asm__ __volatile__(
                       "maskand m0, m1, m1\n"
                       "flw.ps %[d], %[src]\n"
                       "fsw.ps %[d], %[dst]\n"
                       : [ dst ] "=m"(*(uint8_t(*)[32]) dst8), [d] "=&f" (scratch)
                       : [ src ] "m" ( *(const uint8_t(*)[32]) src8)
                       );
  
  src8 += 4*lanes.first;
  dst8 += 4*lanes.first;

  if (DO_EVICTS) {
    evict_va_multi(DO_EVICTS, addr2evict, cl2evict);
  }
  
  if (lanes.second != 0) {
    if (getsize<srcType>() == 2) {
      float o, d;
      __asm__ __volatile__
        (
         "maskand m0, m2, m2\n"
         "flw.ps %[o], %[gatherValues] \n"
         "fgh.ps %[d], %[o](%[src]) \n"
         "fsch.ps %[d], %[o](%[dst]) \n"
         : [d] "=&f" (d), [o] "=&f" (o),
           [ dstMem ] "=m"(*(uint8_t(*)[getsize<srcType>()*8]) dst8)
         : [ srcMem ] "m" ( *(const uint8_t(*)[getsize<srcType>()*8]) src8),
           [ dst ] "r"(dst8),
           [ src ] "r"(src8),
           [ gatherValues ] "m"( * ( const int32_t(*)[8]) gatherValues)
         );
    } else if (getsize<srcType>() == 1) {
      float o, d;
      __asm__ __volatile__
        (
         "maskand m0, m2, m2\n"
         "flw.ps %[o], %[gatherValues] \n"
         "fgb.ps %[d], %[o](%[src]) \n"
         "fscb.ps %[d], %[o](%[dst]) \n"
         : [d] "=&f" (d), [o] "=&f" (o),
           [ dstMem ] "=m"(*(uint8_t(*)[getsize<srcType>()*8]) dst8)
         : [ srcMem ] "m" ( *(const uint8_t(*)[getsize<srcType>()*8]) src8),
           [ dst ] "r"(dst8),
           [ src ] "r"(src8),
           [ gatherValues ] "m" ( * ( const int32_t(*)[8]) gatherValues)
         );
    }
    
    if (DO_EVICTS) {
      fence_evict_va(0, DO_EVICTS, (uintptr_t) dst8, 0);
    }
  }
}

template <ElemKind elK>
INLINE_ATTR void fwdLibInsertTensorInstThreaded(LibTensor* outT, LibTensor* inT, const dim_array_t& coord,
                                                uint32_t count, dim_t axis, uint64_t flags,
                                                const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer();
  void* src = inT->getRawDataPointer();

  dim_t dstDimNum = outT->ndims();
  auto typeSize = static_cast<int32_t>(getsize<srcType>());

  Addresser<elK> tOutput(dst, outT->getScale(), outT->getOffset());
  const Addresser<elK> tAInput(src, inT->getScale(), inT->getOffset());

  const dim_t *dstIndex = outT->dims().data();
  const dim_t* dstPitch = outT->strides().data();
  const dim_t *actIndex = inT->dims().data();
  const dim_t *actPitch = inT->strides().data();

  // Compute virtual pitches of the input, as if it had no padding.
  // This is a trick to avoid bug [SW-11828] when some pitch has been set to 0
  // to make use of the broadcast property.
  dim_array_t actStridesNoPadding = inT->stridesNoPadding();
  const dim_t* actNonPaddingPitch = actStridesNoPadding.data();

  // We compute the offset address: offset in the output tensor where first copy/slice starts (counting is in elements)
  size_t offsetNum = coord[0] * dstPitch[0];
  for (dim_t i = 1; i < dstDimNum; i++) {
    offsetNum += coord[i] * dstPitch[i];
  }

  // Jump between copies in case of count > 1
  size_t jump = dstPitch[axis] * actIndex[axis];

  // Dimension in the output to use as rows, last dim will be processed as chunks
  dim_t dimRow = 0;
  if (dstDimNum > 1) {
    dimRow = dstDimNum - 2;
  }
  dim_t lastDim = dstDimNum - 1;

  int32_t gatherValues[8] = { 0, typeSize, 2 * typeSize, 3 * typeSize,
                              4 * typeSize, 5 * typeSize, 6 * typeSize,
                              7 * typeSize};

  // Taking the number of elements in the input last dimension (contiguous elements), we compute how many lanes will be
  // used and the remainder elements when performing a copy of this last dimension chunk size
  std::pair<int, int> lanes = getLanesResFromNElements<srcType>(static_cast<uint32_t>(actIndex[lastDim]));

  // Setup masks m1 and m2 to enable lanes and remaining elements to be processed per last dim chunk operation
  uint32_t mask = (1 << (((lanes.first - 1) % 8) + 1)) - 1;
  __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                        :
                        : [ mask ] "r" (mask));
  mask = (1 << lanes.second) - 1;
  __asm__ __volatile__("mov.m.x m2, %[mask], 0x0 \n" : : [ mask ] "r"(mask));

  if (axis != lastDim) {
    // Compute number of rows taking into account the count parameter
    size_t auxNRows = count * actIndex[0];
    if (lastDim > 1) {
      auxNRows *= actNonPaddingPitch[0] / actNonPaddingPitch[dimRow];
    }
    // Compute rows per minion and remainder to distribute among the first assigned minions
    size_t mRows = auxNRows / activeMinions;
    size_t mod = auxNRows - activeMinions * mRows;
    if (minionId < mod) {
      ++mRows;
      mod = 0;
    }
    if (unlikely(mRows == 0)) {
      return; // No work to do
    }

    // rows per count operation
    auxNRows /= count;
    // count index where current minion will start
    size_t aux = (mod + mRows * minionId) / auxNRows;
    // jump according to the count index (counting is in elements)
    offsetNum += jump * aux;
    // offset in the input (counting in elements): (total output offset - count times rows per count) times stride
    size_t initialAddrIn = ((mod + mRows * minionId) - aux * auxNRows) * actNonPaddingPitch[dimRow];
    // output start offset (counting is in elements)
    size_t initialAddr = offsetNum;

    // compute vector of coordinates in both input and output tensors for the start position
    dim_t k;
    dim_array_t offsetIn = {0};
    dim_array_t offsetOut = {0};
    getCoordinates(offsetIn, initialAddrIn, dstDimNum, actNonPaddingPitch);
    getNonPaddingCoordinates(offsetOut, initialAddr, dstDimNum, dstPitch, dstIndex, k);

    // translate offsets from elements to elements with padding
    size_t addrOut = 0;
    for (sdim_t i = lastDim; i >= 0; i--) {
      offsetOut[i] += offsetIn[i];
      addrOut += dstPitch[i] * offsetOut[i];
    }
    size_t addrIn = 0;
    for (sdim_t i = lastDim; i >= 0; i--) {
      addrIn += actPitch[i] * offsetIn[i];
    }
    while (mRows > 0) {
      insertRow<srcType>((uint8_t*)dst, (uint8_t*)src, addrOut, addrIn, typeSize, lanes, gatherValues, flags);
      for (sdim_t j = dimRow; j >= 0; j--) {
        if (likely(offsetIn[j] != (actIndex[j] - 1))) {
          // If dimension j is not completed: jump to next element
          addrIn += actPitch[j];
          addrOut += dstPitch[j];
          offsetIn[j]++;
          break;
        } else if (likely(j != 0)) {
          // Reached last element of dimension j (not first dimension): reset that dimension
          addrIn -= (actIndex[j] - 1) * actPitch[j];
          addrOut -= (actIndex[j] - 1) * dstPitch[j];
          offsetIn[j] = 0;
        } else {
          // Reached last element of first dimension: jump to next copy/chunk and start by beginning
          addrIn = offsetIn[j] = 0;
          addrOut += jump - (actIndex[j] - 1) * dstPitch[j];
        }
      }
      mRows--;
    }
  } else {
    // axis == lastDim => this means that we can aggregate count times the chunk copy in the last dimension to be more
    // efficient
    // Compute number or rows *not* taking into account the count parameter
    dim_t auxNRows = actIndex[0];
    if (lastDim > 1) {
      auxNRows *= actNonPaddingPitch[0] / actNonPaddingPitch[dimRow];
    }

    // Compute rows per minion and remainder to distribute among the first assigned minions
    dim_t mRows = auxNRows / activeMinions;
    dim_t mod = auxNRows - activeMinions * mRows;
    size_t initialAddrIn;
    // offset in the input (counting in elements):
    if (minionId < mod) {
      ++mRows;
      initialAddrIn = mRows * actNonPaddingPitch[dimRow] * minionId;
    } else {
      initialAddrIn = (mod + minionId * mRows) * actNonPaddingPitch[dimRow];
    }
    if (unlikely(mRows == 0)) {
      return; // No work to do
    }

    // compute vector of coordinates in both input and output tensors for the start position
    dim_t k;
    dim_array_t offsetIn = {0};
    dim_array_t offsetOut = {0};
    getCoordinates(offsetIn, initialAddrIn, dstDimNum, actNonPaddingPitch);
    getNonPaddingCoordinates(offsetOut, offsetNum, dstDimNum, dstPitch, dstIndex, k);

    // translate offsets from elements to elements with padding
    size_t addrOut = 0;
    for (sdim_t i = lastDim; i >= 0; i--) {
      offsetOut[i] += offsetIn[i];
      addrOut += dstPitch[i] * offsetOut[i];
    }
    size_t addrIn = 0;
    for (sdim_t i = lastDim; i >= 0; i--) {
      addrIn += actPitch[i] * offsetIn[i];
    }

    for (dim_t i = 0; i < mRows; i++) {
      for (dim_t j = 0; j < count; j++) {
        insertRow<srcType>((uint8_t*)dst, (uint8_t*)src, addrOut, addrIn, typeSize, lanes, gatherValues, flags);
        addrOut += jump;
      }
      addrOut -= count * actIndex[axis] * dstPitch[axis];
      for (sdim_t j = dimRow; j >= 0; j--) {
        if (likely(offsetIn[j] != (actIndex[j] - 1))) {
          addrOut += dstPitch[j];
          addrIn += actPitch[j];
          offsetIn[j]++;
          break;
        } else {
          addrOut -= (actIndex[j] - 1) * dstPitch[j];
          addrIn -= (actIndex[j] - 1) * actPitch[j];
          offsetIn[j] = 0;
        }
      }
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _INSERT_TENSOR_INST_H_

