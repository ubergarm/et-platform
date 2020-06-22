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

// This function copies a matrix replacing all the elements which are < splatVal
// and replaces them with splatVal
template <ElemKind elK>
inline void fwdLibMaxSplatInst(LibTensor* outT, LibTensor* inT, float splatVal,
                               uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  if (get_minion_id() != minionOffset) return;

  srcType* const dstT = outT->getRawDataPointer<srcType>();
  srcType* const srcT = inT->getRawDataPointer<srcType>();
  
  Addresser<elK> ptrDstT(dstT, outT->getScale(), outT->getOffset());
  const Addresser<elK> ptrSrcT(srcT, inT->getScale(), inT->getOffset());

  dims_loop<>::run(outT->dims(), outT->strides(), inT->strides(),
                   [&](size_t addrDst, size_t addrSrc) {
                     float src = ptrSrcT[addrSrc];
                     ptrDstT[addrDst] = src > splatVal ? src : splatVal;
                   } );

  
}
template <ElemKind elK>
inline void fwdLibMaxSplatInst(LibTensor* outT, LibTensor* inT, int64_t splatVal,
                               uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  static_assert( elK == Int64ITy);
  if (get_minion_id() != minionOffset) return;

  int64_t* const dstT = outT->getRawDataPointer<int64_t>();
  int64_t* const srcT = inT->getRawDataPointer<int64_t>();
  
  dims_loop<>::run(outT->dims(), outT->strides(), inT->strides(),
                   [&](size_t addrDst, size_t addrSrc) {
                     dstT[addrDst] = (srcT[addrSrc] > splatVal) ? srcT[addrSrc] : splatVal;
                   } );
}

template <ElemKind elK>
inline void fwdLibMaxSplatInstThreaded(LibTensor* outT, LibTensor* inT,
                                       float splatVal, uint64_t flags,
                                       const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;


  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* src = inT->getRawDataPointer<void>();
  
  // Addresser<elK> ptrDstT(dst, scale[1], offset[1]);
  Addresser<elK> ptrDstT(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> ptrSrcT(src, scale[0], offset[0]);
  const Addresser<elK> ptrSrcT(src, inT->getScale(), inT->getOffset());
  
  // unsigned int *actIndex = (unsigned int *)srcDims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  const dim_t *actPitch = inT->strides().data();

  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
    
  unsigned int numElemsDst =
      dstPitch[0] * actIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position

  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k = 0;              // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    float src = ptrSrcT[offsetIn];
    ptrDstT[offsetOut] = (src > splatVal) ? src : splatVal;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <ElemKind elK>
inline void fwdLibMaxSplatInstThreaded(LibTensor* outT, LibTensor* inT,
                                       int64_t splatVal, uint64_t flags,
                                       const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;


  /* maintain compatibility through the new Iface Libtensor */
  void *dst = outT->getRawDataPointer<void>();
  
  // srcType *ptrDstT = (srcType *)dst;
  srcType *ptrDstT = outT->getRawDataPointer<srcType>();
  // srcType *ptrSrcT = (srcType *)src;
  srcType *ptrSrcT = inT->getRawDataPointer<srcType>();
  
  // unsigned int *actIndex = (unsigned int *)srcDims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  const dim_t *actPitch = inT->strides().data();
  
  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  unsigned int numElemsDst =
      dstPitch[0] * actIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position
  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k;                  // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    int64_t src = ptrSrcT[offsetIn];
    ptrDstT[offsetOut] = (src > splatVal) ? src : splatVal;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value, std::size_t>::type = 0,
          bool aligned32B = false>
inline void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, const float *scale, const int32_t *offset){
  float op0, op1;
  // note: no difference between aligned and non aligned versions, as it is not using gathers or scatters
  __asm__ __volatile__(
                       "flw.ps %[op0], %[src]\n"
                       "fbcx.ps %[op1], %[splatVal]\n"
                       "fmax.ps %[op0], %[op0], %[op1]\n"
                       "fsw.ps  %[op0], %[dst] \n"

                       : [op0] "=&f" (op0), [op1] "=&f" (op1),
                         [dst] "=m" (*( char(*)[32]) dst)
                       : [ splatVal ] "r"(bitwise_copy<uint32_t>(splatVal)),
                         [ src] "m" (*(const char(*)[32]) src)
                       );
}


template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, std::size_t>::type = 0,
          bool aligned32B = false >
inline void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, const float *scale, const int32_t *offset){
  // aligned used gather/scatter32, non aligned uses regular ones
  int32_t gatherValues[] = {0, 2, 4, 6, 8, 10, 12, 14};
  float gv, op0, op1;
  __asm__ __volatile__(
                       ".if %[aligned]\n"
                       "    li %[gv], %[gs32_offsets]\n"
                       "    fg32h.ps %[op0],  %[gv](%[src])\n"
                       ".else\n"
                       "    flw.ps %[gv], %[gatherValues]\n"
                       "    fgh.ps %[op0], %[gv](%[src])\n"                       
                       ".endif\n"
                       "fcvt.ps.f16 %[op0], %[op0]\n"
                       "fbcx.ps %[op1], %[splatVal]\n"                       
                       "fmax.ps %[op0], %[op0], %[op1]\n"
                       "fcvt.f16.ps %[op0], %[op0]\n"
                       ".if %[aligned]\n"
                       "    fsc32h.ps %[op0], %[gv](%[dst]) \n"
                       ".else\n"
                       "    fsch.ps  %[op0], %[gv](%[dst]) \n"
                       ".endif\n"

                       : [gv] "=&f" (gv),  [op0] "=&f" (op0), [op1] "=&f" (op1),
                         [dstMem] "=m" (*( char(*)[16]) dst)
                       : [gatherValues ] "m"( *(const int32_t(*)[8]) gatherValues),
                         [splatVal ] "r"(bitwise_copy<uint32_t>(splatVal)),
                         [dst ] "r"(dst),
                         [src ] "r"(src),
                         [srcMem] "m" (*(const char(*)[16]) src),
                         [aligned] "i" (aligned32B),
                         [gs32_offsets] "i" (  fg32h_conf )
                       );
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value,std::size_t>::type = 0,
          bool aligned32B = false > 
inline void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, const float *scale, const int32_t *offset ){
  // aligned used gather/scatter32, non aligned uses regular ones
  float gv, scale_v, offset_v, op0, op1;
  int32_t gatherValues[] = {0, 1, 2, 3, 4, 5, 6, 7};
  __asm__ __volatile__(
                       ".if %[aligned]\n"
                       "    li %[gv], %[gs32_offsets]\n"
                       "    fg32b.ps %[op0], %[gv](%[src])\n"
                       ".else\n"
                       "    flw.ps %[gv], %[gatherValues]\n"
                       "    fgb.ps %[op0], %[gv](%[src])\n"
                       ".endif\n"
                       "fbcx.ps %[offset], %[offset_s] \n"
                       "fbcx.ps %[scale], %[scale_s] \n"
                       "fsub.pi %[op0], %[op0], %[offset] \n"
                       "fcvt.ps.pw %[op0], %[op0] \n"
                       "fmul.ps %[op0], %[op0], %[scale] \n"
                       "fbcx.ps %[op1], %[splatVal]\n"
                       "fmax.ps %[op0], %[op0], %[op1]\n"
                       "frcp.ps %[scale], %[scale] \n"
                       "fcvt.ps.pw %[offset], %[offset] \n"
                       "fmadd.ps %[op0], %[op0], %[scale], %[offset] \n"
                       "fcvt.pw.ps %[op0], %[op0] \n"
                       "fsat8.pi %[op0], %[op0] \n"
                       ".if %[aligned]\n"
                       "    fsc32b.ps %[op0],  %[gv](%[dst]) \n"
                       ".else\n"
                       "    fscb.ps  %[op0], %[gv](%[dst]) \n"
                       ".endif\n"
                       : [gv] "=&f" (gv), [op0] "=&f" (op0), [op1] "=&f" (op1),
                         [scale] "=&f" (scale_v), [offset] "=&f" (offset_v),
                         [dstMem] "=m" (*( char(*)[8]) dst)
                       : [gatherValues ] "m"( *(const int32_t(*)[8]) gatherValues),
                         [dst ] "r"(dst),
                         [src ] "r"(src),
                         [splatVal ] "r"(bitwise_copy<uint32_t>(splatVal)),
                         [scale_s ] "r"(bitwise_copy<uint32_t>(scale[0])),
                         [offset_s ] "r"(offset[0]),
                         [srcMem] "m" (*(const char(*)[8]) src),
                         [aligned] "i" (aligned32B),
                         [gs32_offsets] "i" (  fg32b_conf )
                        );
  //FIXME:  only using scale[0] and offset[0] => probably something wrong here;
  
}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value
                                                    && !std::is_same<srcType, float16>::value
                                                    && !std::is_same<srcType, float>::value, std::size_t>::type = 0,
          bool aligned32B = false>
inline void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, const float *scale, const int32_t *offset ){
  //FIXME: implementation missing
}

template <ElemKind elK>
inline void fwdLibMaxSplatInstVectorized(LibTensor* outT, LibTensor* inT,
                                         float splatVal, uint64_t flags,
                                         const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* src = inT->getRawDataPointer<void>();

  uintptr_t dstAddr = (uintptr_t)dst;
  uintptr_t srcAddr = (uintptr_t)src;

  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)srcDims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  const dim_t *actPitch = inT->strides().data();
 
  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  unsigned int numElemsDst = dstPitch[0] * actIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;
  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  unsigned int elementsInRow, registersInRow, res;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

  while (!done && (offsetOut < posMax)) {
  if (firstRow && coord[lastDim - 1] != maxRow) {
      elementsInRow = dstIndex[lastDim] - coord[lastDim];
    } else if (coord[lastDim - 1] == maxRow) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
    } else {
      elementsInRow = dstIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      res = elementsInRow - registersInRow * 8;
      mask = ((1 << res) - 1);
      __asm__ __volatile__("mov.m.x m1, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr += offsetIn * typeSize;
    dstAddr += offsetOut * typeSize;

    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
    const float scale[2] = {inT->getScale(), outT->getScale()};
    const int32_t offset[2] = {inT->getOffset(), outT->getOffset()}; 
    for (unsigned int i = 0; i < registersInRow; i++) {
      maxSplatOp <srcType>(dstAddr, srcAddr, splatVal, scale, offset);
      srcAddr += 8 * typeSize;
      dstAddr += 8 * typeSize;
    }

    if (res > 0) {
      __asm__ __volatile__("maskand m0, m1, m0 \n");
      maxSplatOp <srcType>(dstAddr, srcAddr, splatVal, scale, offset);
    }

    if (lastRow)
      return;

    dstAddr = (uintptr_t)dst;
    srcAddr = (uintptr_t)src;
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;


  done = getOffsets(lastDim, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <ElemKind elK>
inline void fwdLibMaxSplatInstAligned32Bytes(LibTensor* outT, LibTensor* inT,
                                             float splatVal, uint64_t flags,
                                             const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  uintptr_t dstAddr;
  uintptr_t srcAddr;

  /* maintain compatibility through the new Iface Libtensor */
  void *dst = outT->getRawDataPointer<void>();
  void *src = inT->getRawDataPointer<void>();
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)srcDims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  const dim_t *actPitch = inT->strides().data();

  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  unsigned int numElemsDst = dstPitch[0] * actIndex[0];
  unsigned int initialAddr, maxRead;
  
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }


  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  unsigned int n_dstPitch[outT->ndims()];
  unsigned int n_dstIndex[outT->ndims()];
  unsigned int n_actIndex[inT->ndims()];
  unsigned int n_actPitch[inT->ndims()];
  
  for(size_t i = 0; i < inT->ndims(); i++) {
    n_actPitch[i] = actPitch[i];
    n_actIndex[i] = actIndex[i];
    n_dstPitch[i] = dstPitch[i];
    n_dstIndex[i] = dstIndex[i];    
  }

  unsigned int lastDim = srcDimNum - 1;
  unsigned int res = ((n_dstIndex[lastDim] - 1)%8) +1;
  n_actPitch[lastDim] *= 8;
  n_dstPitch[lastDim] *= 8;
  n_dstIndex[lastDim] = (n_dstIndex[lastDim] - 1)/8 + 1;
  unsigned int mask = ((1 << res) - 1);

  const float scale[2] = {inT->getScale(), outT->getScale()};
  const int32_t offset[2] = {inT->getOffset(), outT->getOffset()}; 

  while (!done && (offsetOut < posMax)) {
    dstAddr = (uintptr_t)dst + offsetOut*typeSize;
    srcAddr = (uintptr_t)src + offsetIn*typeSize;

    if (coord[lastDim] != dstIndex[lastDim] - 1)
         __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
    else __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);

    maxSplatOp <srcType, true>(dstAddr, srcAddr, splatVal, scale, offset);

    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, n_actIndex,
                      n_actPitch, n_dstPitch);
  }
  if (DO_EVICTS) {
    unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
    if (clperminion > 0)
      fence_evict_va(0, DO_EVICTS, initialAddr, clperminion - 1, CACHE_LINE_BYTES);
  }
}



template <ElemKind elK, typename splatval_t>
inline void fwdLibMaxSplatInstBest(const int desired, LibTensor* outT, LibTensor* inT, splatval_t splatVal,
                               uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

 switch(desired){
 case 1: fwdLibMaxSplatInst<elK>(outT, inT, splatVal, flags, minionOffset, assignedMinions); break;
 case 2: fwdLibMaxSplatInstThreaded<elK>(outT, inT, splatVal, flags, minionOffset, assignedMinions); break;
 case 3: fwdLibMaxSplatInstVectorized<elK>(outT, inT, splatVal, flags, minionOffset, assignedMinions); break;      
 default:
   {
     const size_t batchDim = inT->ndims() - 2;
     if (inT->ndims() >= 2 &&
         ( outT->strides()[batchDim] % 32 == 0 ||  32 % outT->strides()[batchDim] == 0 ) &&
         (  inT->strides()[batchDim] % 32 == 0 ||  32 %  inT->strides()[batchDim] == 0 ))
       fwdLibMaxSplatInstAligned32Bytes<elK>(outT, inT, splatVal, flags, minionOffset, assignedMinions);
     else
       fwdLibMaxSplatInstVectorized<elK>(outT, inT, splatVal, flags, minionOffset, assignedMinions);
   }
   break;
 }
 
}
 
} // namespace inlining

} // namespace dnn_lib

#endif // __MAX_SPLAT_INST_H_
