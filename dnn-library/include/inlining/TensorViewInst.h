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

#ifndef _TENSOR_VIEW_INST_H_
#define _TENSOR_VIEW_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>
#include <utility>

#include "Float16.h"
#include "Writer.h" // From include/internal path
#include "Addresser.h" // From include/internal path
#include "Converter.h" // From include/internal path
#include "Operator.h" // From include/internal path
#include "utils.h" // From include/internal path
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

inline __attribute__((always_inline))
  unsigned int minTview(uint8_t &type, unsigned int a, unsigned int b,
                        unsigned int c) {
  type = 0;
  if(b < a) {
    type = 1;
    a = b;
  }
  if(c < a) {
    type = 2;
    return c;
  }
  return a;
}

template <typename srcType>
inline __attribute__((always_inline)) void
gatherScatterTView(uint8_t *src8, uint8_t *dst8, const uint32_t &mask,
                   int32_t *gatherValues) {
  float d0,d1;
  if (sizeof(srcType) == 2) {
    __asm__ __volatile__
      (
       "mov.m.x m0, %[mask], 0\n"
       "flw.ps %[d1], %[gatherValues] \n"
       "fgh.ps %[d0], %[d1](%[src]) \n"
       "fsch.ps %[d0], %[d1](%[dst]) \n"
       : [dstMem] "=m" (*(uint8_t(*)[16]) dst8),
         [d0] "=&f" (d0), [d1] "=&f" (d1)
       : [ src ] "r"(src8), [ dst ] "r"(dst8),
         [srcMem] "m" (*(const uint8_t(*)[16]) src8),
         [ gatherValues ] "m"( *(const int32_t(*)[8]) gatherValues),
         [ mask ] "r" (mask)
      );

  } else if (sizeof(srcType) == 1) {
    __asm__ __volatile__(
        "mov.m.x m0, %[mask], 0\n"
        "flw.ps %[d1], %[gatherValues] \n"
        "fgb.ps %[d0], %[d1](%[src]) \n"
        "fscb.ps %[d0], %[d1](%[dst]) \n"
        : [dstMem] "=m" (*(uint8_t(*)[8]) dst8),
          [d0] "=&f" (d0), [d1] "=&f" (d1)
        : [ src ] "r"(src8), [ dst ] "r"(dst8),
          [srcMem] "m" (*(const uint8_t(*)[8]) src8),
          [ gatherValues ] "m"( *(const int32_t(*)[8]) gatherValues),
          [ mask ] "r" (mask)
     );
  }
  return;
}



  // just one generic and vectorized version
template <ElemKind elK>
inline void fwdLibTensorViewInst(LibTensor* outT, LibTensor* inT,
                                           const dim_array_t & coord, uint64_t flags,
                                           const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */

  void *dst = outT->getRawDataPointer<void>();
  void *src = inT->getRawDataPointer<void>();
  
  // Addresser<elK> tOutput(dst, scale[1], offset[1]);
  Addresser<elK> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tAInput(src, scale[0], offset[0]);
  const Addresser<elK> tAInput(src, inT->getScale(), inT->getOffset());

  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)srcDims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  const dim_t *actPitch = inT->strides().data();
  
  unsigned int dstDimNum = static_cast<unsigned int>(outT->ndims());
  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddrOut, maxRead;
  int32_t typeSize = (int32_t) sizeof(srcType);
  getCachelinePartition(typeSize, numElemsDst, initialAddrOut, maxRead,
                        minionId, activeMinions, dst); // Obtain initial addr 4 minions
  if (maxRead == 0)
    return; // No work to do for the minion

  unsigned int coordIn[srcDimNum], coordOut[dstDimNum], kOut;
  getNonPaddingCoordinates(coordOut, initialAddrOut, dstDimNum, dstPitch,
                           dstIndex, kOut); // Find the first useful coord vect
  unsigned int srcLastDim = srcDimNum - 1;
  unsigned int dstLastDim = dstDimNum - 1;
  unsigned int auxActPitch[srcDimNum], auxdstPitch[dstDimNum];
  auxActPitch[srcLastDim] = auxdstPitch[dstLastDim] = 1;
  for (int i = srcDimNum - 2; i >= 0; i--)
    auxActPitch[i] = auxActPitch[i + 1] * actIndex[i + 1];
  for (int i = dstDimNum - 2; i >= 0; i--)
    auxdstPitch[i] = auxdstPitch[i + 1] * dstIndex[i + 1];

  unsigned int addrIn, addrOut, elements_moved;
  addrIn = addrOut = elements_moved = 0;

   for (unsigned int j = 0; j < kOut; j++) { // Compute the output address
    addrOut += dstPitch[j] * coordOut[j];
    elements_moved += auxdstPitch[j] * coordOut[j];
  }
  for (unsigned int i = 0; i < srcDimNum; i++) { // Compute the input coord vec
    coordIn[i] = elements_moved / auxActPitch[i];
    elements_moved = elements_moved - coordIn[i] * auxActPitch[i];
  }
  for (int i = srcLastDim; i >= 0; i--) { // Add to in coord vect the offset
    coordIn[i] += (int)coord[i];
    if (coordIn[i] >= actIndex[i]) {
      coordIn[i] = coordIn[i] % actIndex[i];
      coordIn[i - 1] += 1;
    }
    addrIn += coordIn[i] * actPitch[i]; // Compute input address
  }
  uint8_t *dst8 = (uint8_t *) dst + addrOut*typeSize;;
  uint8_t *src8 = (uint8_t *) src + addrIn*typeSize;
  unsigned int posMax = std::min(maxRead + initialAddrOut, numElemsDst); // Last position to "copy"
  maxRead = posMax - addrOut;

  int32_t gatherValues[8] = { 0, typeSize, 2 * typeSize, 3 * typeSize,
                               4 * typeSize, 5 * typeSize, 6 * typeSize,
                               7 * typeSize}; //Computed at compilation time

  bool done = false;
  while ((addrOut < posMax) & !done) {
    uint8_t type;
    unsigned int d = minTview(type, maxRead,
                               actIndex[srcLastDim] - coordIn[srcLastDim],
                               dstIndex[dstLastDim] - coordOut[dstLastDim]);
    if(type == 1) {
      addrOut += (d - 1) * dstPitch[dstLastDim];
      coordOut[dstLastDim] += (d - 1);
      addrIn -= coordIn[srcLastDim] * actPitch[srcLastDim];
      coordIn[srcLastDim] = 0;
    }
    else if (type == 2) {
      addrIn += (d - 1) * actPitch[srcLastDim];
      coordIn[srcLastDim] += (d - 1);
      addrOut -= coordOut[dstLastDim] * dstPitch[dstLastDim];
      coordOut[dstLastDim] = 0;
    }

    maxRead -= d; // FIXME it does not support doubles

    std::pair<int, int> lanes = getLanesResFromNElements<srcType>(d);

    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n");
    while (lanes.first >= 8) {
      __asm__ __volatile__("flw.ps f0, 0x0(%[src])\n"
                           "fsw.ps f0, 0x0(%[dst])\n"
                           :
                           : [ src ] "r"(src8), [ dst ] "r"(dst8)
                           : "f0", "memory");
      lanes.first -= 8;
      src8 += 32;
      dst8 += 32;
    }
    if (lanes.first != 0) {
      uint32_t mask = ((1 << lanes.first) - 1);
      __asm__ __volatile__(
        "mov.m.x m0, %[mask], 0\n"
        "flw.ps f0, 0x0(%[src])\n"
        "fsw.ps f0, 0x0(%[dst])\n"
        :
        : [ src ] "r"(src8), [ dst ] "r"(dst8),
          [ mask ] "r" (mask)
        : "f0", "memory");
      src8 += 4*lanes.first;
      dst8 += 4*lanes.first;
    }
    if (lanes.second != 0) {
      uint8_t mask = ((1 << lanes.second) - 1);
      gatherScatterTView <srcType>(src8, dst8, mask, gatherValues);
    }
    if (type == 0)
      break;
    if (type == 1) {
      done = getOffsets(srcLastDim, coordIn, addrIn, actIndex, actPitch);
      done = getOffsets(dstDimNum, coordOut, addrOut, dstIndex, dstPitch);
    }
    else if(type == 2) {
      done = getOffsets(srcDimNum, coordIn, addrIn, actIndex, actPitch);
      done = getOffsets(dstLastDim, coordOut, addrOut, dstIndex, dstPitch);
      // TODO this last getOffsets could be avoided, as in lines 4144, 4145
      // we could use d instead of d - 1. This yields no problems as in
      // the min function type == 2 iff the inequality is strict
    }
    src8 = (uint8_t *) src + typeSize * addrIn;
    dst8 = (uint8_t *) dst + typeSize * addrOut;
  }

  if (!DO_EVICTS) // Evicting the result
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddrOut, clperminion);
}

} // namespace inlining

} // namespace dnn_lib


#endif // _TENSOR_VIEW_INST_H_
