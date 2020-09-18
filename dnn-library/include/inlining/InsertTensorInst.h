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
#include "cacheops.h"
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

  template <ElemKind elK>
inline __attribute__((always_inline))
  void fwdLibInsertTensorInst(LibTensor* outT, LibTensor* inT, const dim_array_t offsets,
                            unsigned int count, unsigned int axis,
                            uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  if (get_minion_id() != minionOffset) return;
  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* src = inT->getRawDataPointer<void>();

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
inline void insertRow(uint8_t *dst, uint8_t *src, const unsigned int& addrOut,
                      const unsigned int& addrIn, const int32_t& typeSize,
                      std::pair<int, int> lanes, int32_t *gatherValues, uint64_t flags) 
{
  uint8_t *dst8 = (uint8_t *) dst + addrOut * typeSize;
  uint8_t *src8 = (uint8_t *) src + addrIn * typeSize;
  float scratch;

  uintptr_t addr2evict = (uintptr_t)dst8;
  // Computes bytes to evict (adds the unaligned cache line bytes of the base address
  uint32_t  bytes2evict = ((lanes.first * 4) + (addr2evict & 0x3F));
  // With that computes the cl2evict
  uint32_t  cl2evict = (bytes2evict + CACHE_LINE_BYTES -1)/ CACHE_LINE_BYTES;

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
inline __attribute__((always_inline))
void fwdLibInsertTensorInstThreaded(LibTensor* outT, LibTensor* inT,
                                    const dim_array_t &coord,
                                    unsigned int count,
                                    unsigned int axis, uint64_t flags,
                                    const uint32_t minionOffset = 0,
                                    const  uint32_t assignedMinions = 0) {

  using srcType = typename elemKind2elemTy<elK>::type;
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) :  assignedMinions;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* src = inT->getRawDataPointer<void>();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();

  unsigned int dstDimNum = static_cast<unsigned int>(outT->ndims());
   
  int32_t typeSize = (int32_t) getsize<srcType>();

  
  // Addresser<elK> tOutput(dst, scale[1], offset[1]);
  Addresser<elK> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tAInput(src2, scale[0], offset[0]);
  const Addresser<elK> tAInput(src, inT->getScale(), inT->getOffset());

  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)src2Dims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *actPitch = (unsigned int *)src2Pitches;
  const dim_t *actPitch = inT->strides().data();
 
  // We compute the offset address
  unsigned int offsetNum = coord[0] * dstPitch[0];
  for (unsigned int i = 1; i < dstDimNum; i++)
    offsetNum += coord[i] * dstPitch[i]; // Offset Address
  unsigned int jump = dstPitch[axis] * actIndex[axis];

  unsigned int dimRow = 0;
  if (dstDimNum > 1)
    dimRow = dstDimNum - 2;
  unsigned int lastDim = dstDimNum - 1;

  int32_t gatherValues[8] = { 0, typeSize, 2 * typeSize, 3 * typeSize,
                              4 * typeSize, 5 * typeSize, 6 * typeSize,
                              7 * typeSize};

  std::pair<int, int> lanes = getLanesResFromNElements<srcType>(actIndex[lastDim]);

  uint32_t mask = (1 << (((lanes.first - 1) % 8) + 1)) - 1;
  __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                        :
                        : [ mask ] "r" (mask));
  mask = (1 << lanes.second) - 1;
  __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                        :
                        : [ mask ] "r" (mask));

  if (axis != lastDim) {
    unsigned int auxNRows = count * actIndex[0];
    for (unsigned i = 1; i < lastDim; i++)
      auxNRows *= actIndex[i];
    unsigned int mRows = auxNRows / activeMinions;
    unsigned int mod = auxNRows - activeMinions * mRows;
    if (minionId < mod) {
      ++mRows;
      mod = 0;
    }
    if (unlikely(mRows == 0))
      return; // No work to do

    auxNRows /= count;
    unsigned int aux = (mod + mRows * minionId) / auxNRows;
    offsetNum += jump * aux;
    unsigned int initialAddrIn = ((mod + mRows * minionId) - aux * auxNRows) * actPitch[dimRow];

    unsigned int offsetIn[dstDimNum], offsetOut[dstDimNum];
    unsigned int initialAddr = offsetNum;
    getCoordinates(offsetIn, initialAddrIn, dstDimNum, actPitch);
    getCoordinates(offsetOut, initialAddr, dstDimNum, dstPitch);

    unsigned int addrOut = 0;
    for (int i = lastDim; i >= 0; i--) {
      offsetOut[i] += offsetIn[i];
      addrOut += dstPitch[i] * offsetOut[i];
    }
    while (mRows > 0) {
      insertRow<srcType>((uint8_t *) dst, (uint8_t *) src, addrOut,
                         initialAddrIn, typeSize, lanes, gatherValues, flags);
      for (int j = dimRow; j >= 0; j--) {
        if (likely(offsetIn[j] != (actIndex[j] - 1))) {
          initialAddrIn += actPitch[j];
          addrOut += dstPitch[j];
          offsetIn[j]++;
          break;
        } else if (likely(j != 0)){
          initialAddrIn -= (actIndex[j] - 1) * actPitch[j];
          addrOut -= (actIndex[j] - 1) * dstPitch[j];
          offsetIn[j] = 0;
        } else {
          initialAddrIn = offsetIn[j]  = 0;
          offsetNum += jump;
          addrOut = offsetNum;
        }
      }
      mRows--;
    }
  } else {
    unsigned int auxNRows = actIndex[0];
    for (unsigned i = 1; i < dstDimNum - 1; i++)
      auxNRows *= actIndex[i];

    if (auxNRows > activeMinions) {
      unsigned int mRows = auxNRows / activeMinions;
      unsigned int mod = auxNRows - activeMinions * mRows;
      unsigned int initialAddrIn;
      // We add to the initial address the new address in the tensor
      if (minionId < mod) {
        ++mRows;
        initialAddrIn = mRows * actPitch[dimRow] * minionId;
      } else
        initialAddrIn = (mod + minionId * mRows) * actPitch[dimRow];
      unsigned int k, offsetIn[dstDimNum], offsetOut[dstDimNum];
      getNonPaddingCoordinates(offsetIn, initialAddrIn, dstDimNum, actPitch,
                               actIndex, k);
      getNonPaddingCoordinates(offsetOut, offsetNum, dstDimNum, dstPitch,
                               dstIndex, k);
      unsigned int addrOut = 0;
      for (int i = dstDimNum - 1; i >= 0; i--) {
        offsetOut[i] += offsetIn[i];
        addrOut += dstPitch[i] * offsetOut[i];
      }
      for (unsigned i = 0; i < mRows; i++) {
        for (unsigned j = 0; j < count; j++) {
          insertRow<srcType>((uint8_t *) dst, (uint8_t *) src, addrOut,
                             initialAddrIn, typeSize, lanes, gatherValues, flags);
          addrOut += actIndex[axis] * dstPitch[axis];
        }
        addrOut -= count * actIndex[axis] * dstPitch[axis];
        for (int j = dimRow; j >= 0; j--) {
          if (offsetIn[j] != (actIndex[j] - 1)) {
            addrOut += dstPitch[j];
            initialAddrIn += actPitch[j];
            offsetIn[j]++;
            break;
          } else {
            addrOut -= (actIndex[j] - 1) * dstPitch[j];
            initialAddrIn -= (actIndex[j] - 1) * actPitch[j];
            offsetIn[j] = 0;
          }
        }
      }
    } else {
      unsigned int mperRow = activeMinions / auxNRows;
      if (minionId >= mperRow * auxNRows)
        return;
      unsigned int rowtomin = minionId / mperRow;

      unsigned int offsetOut[dstDimNum];
      for (unsigned int i = 0; i < dstDimNum; i++) {
        offsetOut[i] = coord[i];
      }

      if(axis > 0) {
        unsigned int falsepitch[axis];
        falsepitch[dimRow] = 1;
        for (int i = dimRow; i > 0; i--)
          falsepitch[i - 1] = falsepitch[i] * actIndex[i];

        for (unsigned i = 0; i < axis; i++) {
          unsigned int aux = rowtomin / falsepitch[i];
          offsetOut[i] += aux;
          rowtomin -= aux * falsepitch[i];
        }
      }
      unsigned int addrOut = 0;
      for (int i = axis; i >= 0; i--) {
        addrOut += dstPitch[i] * offsetOut[i];
      }
      unsigned int lastRowElem = addrOut + actIndex[axis] * dstPitch[axis] * count;
      unsigned int cll = CACHE_LINE_BYTES / getsize<srcType>();
      unsigned int modulo = addrOut % cll;
      //unsigned int maximalPos = jump * count;
      unsigned int clperRow = (modulo + (jump * count) - 1) / cll + 1;
      unsigned int mcl = clperRow / mperRow;
      unsigned int mod = clperRow - mperRow * mcl;
      unsigned int maxRead;
      unsigned int minmodule = minionId % mperRow;
      if (minmodule != 0) {
        addrOut -= modulo;
        if (minmodule < mod){
          ++mcl;
          addrOut += mcl * cll * minmodule;
        } else {
          addrOut += (mod + minmodule * mcl) * cll;
        }
        maxRead = mcl * cll;
      } else {
        if (mod != 0) {
          ++mcl;
        }
        maxRead = mcl * cll - modulo;
      }
      if (mcl == 0) {
        return;
      }
      //maximalPos += save - 1;
      unsigned int k;
      getNonPaddingCoordinates(offsetOut, addrOut, dstDimNum, dstPitch,
                               dstIndex, k);
      addrOut = 0;
      for (unsigned int i = 0; i < dstDimNum; i++) {
        addrOut += offsetOut[i] * dstPitch[i];
      }

      unsigned int offsetIn[dstDimNum];
      for (unsigned int i = 0; i < dstDimNum; i++) {
        offsetIn[i] = offsetOut[i] - coord[i];
      }
      offsetIn[axis] = offsetIn[axis] % actIndex[axis];
      unsigned int initialAddrIn = 0;
      for (unsigned int i = 0; i < dstDimNum; i++) {
        initialAddrIn += offsetIn[i] * actPitch[i];
      }
      maxRead = std::min(maxRead, lastRowElem - addrOut);
      unsigned int length = std::min(static_cast<dim_t>(maxRead), actIndex[axis] - offsetIn[axis]);

      std::pair<int, int> auxlanes = getLanesResFromNElements<srcType>(length);
      maxRead -= length;
      mask = (1 << (((auxlanes.first - 1) % 8) + 1)) - 1;
      __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      mask = (1 << auxlanes.second) - 1;
      __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));

      insertRow<srcType>((uint8_t *) dst, (uint8_t *) src, addrOut,
                         initialAddrIn, typeSize, auxlanes, gatherValues, flags);
      addrOut += length * dstPitch[axis];
      initialAddrIn -= offsetIn[axis] * actPitch[axis];

      mask = (1 << (((lanes.first - 1) % 8) + 1)) - 1;
      __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      mask = (1 << lanes.second) - 1;
      __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      while (maxRead > actIndex[axis]) {
        insertRow<srcType>((uint8_t *) dst, (uint8_t *) src, addrOut,
                           initialAddrIn, typeSize, lanes, gatherValues,flags);
        maxRead -= actIndex[axis];
        addrOut += actIndex[axis] * dstPitch[axis];
      }

      auxlanes = getLanesResFromNElements<srcType>(maxRead);

      mask = (1 << (((auxlanes.first - 1) % 8) + 1)) - 1;
      __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      mask = (1 << auxlanes.second) - 1;
      __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      insertRow<srcType>((uint8_t *) dst, (uint8_t *) src, addrOut,
                         initialAddrIn, typeSize, auxlanes, gatherValues, flags);
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _INSERT_TENSOR_INST_H_

