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

#ifndef _TRANSPOSE_INST_H_
#define _TRANSPOSE_INST_H_

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


template <typename srcType, typename std::enable_if< (sizeof(srcType) <=4), int>::type = 0>
void transposeOp (uintptr_t dst, uintptr_t src, int32_t *scatterValues, int32_t *gatherValues){
  __asm__ __volatile__("flw.ps f31, %[gatherValues] \n"
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
                       : [ gatherValues ] "m"( *(const int32_t(*)[8]) gatherValues),
                         [ scatterValues ] "m"(*(const int32_t(*)[8]) scatterValues),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src),
                         [ size] "i" (sizeof(srcType))
                       : "f0", "f31", "memory");
}

template <typename srcType, typename std::enable_if< (sizeof(srcType) > 4), int>::type = 0 >
void transposeOp (uintptr_t dst, uintptr_t src, int32_t *scatterValues,  int32_t *gatherValues){
  //FIXME: TODO: implement
}



  // Vectorized version is the generic
  template <ElemKind elK, size_t N>
inline void fwdLibTransposeInst(LibTensor* outT, LibTensor* inT,
                                          const std::array<uint32_t, N> &shuffle, uint64_t flags,
                                          const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void *dst = outT->getRawDataPointer<void *>();
  void *src = inT->getRawDataPointer<void *>();
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  const dim_t *actPitch = inT->strides().data();

  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = sizeof(srcType);
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dst);
  if (maxRead == 0)
    return;

  unsigned int newPitch[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    newPitch[i] = actPitch[shuffle[i]];

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex, k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += newPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;

  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

  unsigned int newPitchSize = newPitch[lastDim] * typeSize;
  int32_t gatherValues[8];
  for (unsigned int i = 0; i < 8; i++) gatherValues[i] = i*newPitchSize;
  unsigned int dstPitchSize = dstPitch[lastDim] * typeSize;
  int32_t scatterValues[8];
  for (unsigned int i = 0; i < 8; i++) scatterValues[i] = i*dstPitchSize;

  // Work pending to be done
  while (!done && (offsetOut < posMax)) {
    // Compute number of elements in current row
    int elementsInRow = dstIndex[lastDim] - coord[lastDim];
    if ((offsetOut + elementsInRow) > posMax) {
      elementsInRow = posMax - offsetOut;
    }

    // Starting addresses
    uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src) + offsetIn * typeSize;
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dst) + offsetOut * typeSize;

    // Computes full passes and partial passes
    int registersInRow = elementsInRow / 8;
    int res = elementsInRow - registersInRow * 8;
    
    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n");
    for (int i = 0; i < registersInRow; i++) {
      transposeOp <srcType>(dstAddr, srcAddr, scatterValues, gatherValues);
      srcAddr += 8 * typeSize * newPitch[lastDim];
      dstAddr += 8 * typeSize;
    }

    if (res > 0) {
      uint8_t mask = ((1 << res) - 1);
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
  unsigned int clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr,
                                      clperminion);
}


template <typename srcType, typename std::enable_if< (sizeof(srcType) <=4), int>::type = 0 >
void transposeOpAligned32Bytes (uintptr_t dst, uintptr_t src, int32_t *gatherValues){
  constexpr size_t size = sizeof(srcType);
  constexpr auto g32_conf = size == 2 ? fg32h_conf : fg32b_conf;
  
  __asm__ __volatile__("flw.ps f31, %[gatherValues] \n"
                       ".if %[size] == 4\n"
                       "    fgw.ps  f0, f31(%[src]) \n"
                       "    fsw.ps  f0, (%[dst]) \n"
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
                       : [ gatherValues ] "m"( *(const int32_t(*)[8]) gatherValues),
                         [ src ] "r"(src),
                         [ dst ] "r"(dst),
                         [g32_conf] "i" (g32_conf),
                         [size] "i" (size)
                       : "f0", "f31", "t0", "memory"
                       );
}
  
template <typename srcType, typename std::enable_if< (sizeof(srcType) > 4), int>::type = 0>
  void transposeOpAligned32Bytes (uintptr_t dst, uintptr_t src, int32_t *gatherValues){
  //FIXME: not implemented
}



  template <ElemKind elK, size_t N>
inline void fwdLibTransposeInstAligned32Bytes(LibTensor* outT, LibTensor* inT,
                                              const std::array<uint32_t, N> &shuffle, uint64_t flags,
                                              const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;  

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  void *dst = outT->getRawDataPointer<void>();
  void *src = inT->getRawDataPointer<void>();
  
  // uintptr_t dstAddr = (uintptr_t)dst;
  // uintptr_t srcAddr = (uintptr_t)srcp;
  uintptr_t dstAddr = reinterpret_cast<uintptr_t>(dst);
  uintptr_t srcAddr = reinterpret_cast<uintptr_t>(src);

  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  const dim_t *actPitch = inT->strides().data();
  
  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = sizeof(srcType);
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dst);
  if (maxRead == 0)
    return;

  unsigned int newPitch[srcDimNum];
  unsigned int newdstPitch[srcDimNum];
  unsigned int newdstIndex[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++) {
    newPitch[i] = actPitch[shuffle[i]];
    newdstPitch[i] = dstPitch[i];
    newdstIndex[i] = dstIndex[i];
  }

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += newPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  unsigned int lastDim = srcDimNum - 1;
  unsigned int newPitchSize = newPitch[lastDim] * typeSize;
  int32_t gatherValues[8];
  for (unsigned int i = 0; i < 8; i++) gatherValues[i] = i*newPitchSize;

  //We modify the pitches and coord so that the function getOffsets
  //jumps eight positions in lastDim, the smallest dimension.
  //Number 8 is the amount of lanes that a register has.
  unsigned int res = ((dstIndex[lastDim] - 1)%8) + 1;
  newPitch[lastDim] *= 8;
  newdstPitch[lastDim] *= 8;
  newdstIndex[lastDim] = (dstIndex[lastDim] - 1)/8 + 1;
  unsigned int mask = ((1 << res) - 1);

  while (!done && (offsetOut < posMax)) {
    // dstAddr = (uintptr_t)dst + offsetOut*typeSize;
    dstAddr = reinterpret_cast<uintptr_t>(dst) + offsetOut*typeSize;
    // srcAddr = (uintptr_t)src + offsetIn*typeSize;
    srcAddr = reinterpret_cast<uintptr_t>(src) + offsetIn*typeSize;

    //When the minion reaches the end of the lastDim, we use a mask
    //that is always the same because the dst Tensor is aligned to 32 Bytes.
    if (coord[lastDim] != newdstIndex[lastDim] - 1)
         __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
    else __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);

    transposeOpAligned32Bytes <srcType>(dstAddr, srcAddr, gatherValues);
    done = getOffsets(srcDimNum, coord, offsetOut, offsetIn, newdstIndex, newdstPitch, newPitch);
  }
  if (DO_EVICTS) {
    unsigned int clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
    if (clperminion > 0)
      fence_evict_va(0, DO_EVICTS, initialAddr, clperminion - 1, CACHE_LINE_BYTES);
  }
}


} // namespace inlining

} // namespace dnn_lib

#endif //  _TRANSPOSE_INST_H_
