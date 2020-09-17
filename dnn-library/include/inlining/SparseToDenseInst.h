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


template <ElemKind elK, typename std::enable_if<elK == FloatTy,std::size_t>::type = 0>
inline __attribute__((always_inline)) void sparseToDenseOp (uintptr_t dst, uintptr_t src, long long* tIndex, unsigned int batchPitch,
unsigned int batch, unsigned int numIndices, size_t typeSize, const float *scale, const int32_t *offset){

  __asm__ __volatile__("add t0, zero, zero\n"
                       "fxor.pi f0, f0, f0\n"

                       "addi    t3, %[tIndex], 0x0\n"
                       "1:\n"

                       "ld t1, 0x0(t3)\n"
                       "bne t1, %[batch], 2f\n"

                       "mul t2, t0, %[typeSize]\n"
                       "mul t2, t2, %[batchPitch]\n"
                       "add t2, t2, %[src]\n"
                       "flw.ps  f1, 0(t2) \n"
                       "fadd.ps f0, f0, f1 \n"
                       "2:\n"

                       "addi t0, t0, 0x1\n"
                       "addi t3, t3, 0x8\n"
                       "blt t0, %[numIndices], 1b\n"

                       "fsw.ps  f0, %[dst] \n"

                       : [ dst ] "=m" (* (char(*)[32]) dst)
                       : [ src ] "r"(src),
                         [ numIndices ] "r"(numIndices),
                         [ batch ] "r"(batch),
                         [ tIndex ] "r"(tIndex),
                         [ batchPitch ] "r"(batchPitch),
                         [ typeSize ] "r"(typeSize)
                       : "t0", "t1", "t2", "t3", "f0", "f1", "memory");

}

template <ElemKind elK, typename std::enable_if<elK == Float16Ty,std::size_t>::type = 0>
inline __attribute__((always_inline)) void sparseToDenseOp (uintptr_t dst, uintptr_t src, long long* tIndex, unsigned int batchPitch,
unsigned int batch, unsigned int numIndices, size_t typeSize, const float *scale, const int32_t *offset){
  int32_t gatherValues[] = {0, 2, 4, 6, 8, 10, 12, 14};


  __asm__ __volatile__("add t0, zero, zero\n"
                       "fxor.pi f0, f0, f0\n"
                       "fcvt.ps.f16 f0, f0\n"
                       "addi    t3, %[tIndex], 0x0\n"
                       "flw.ps f31, %[gatherValues]\n"
                       "1:\n"

                       "ld t1, 0x0(t3)\n"
                       "bne t1, %[batch], 2f\n"

                       "mul t2, t0, %[typeSize]\n"
                       "mul t2, t2, %[batchPitch]\n"
                       "add t2, t2, %[src]\n"
                       "fgh.ps  f1, f31(t2)\n"
                       "fcvt.ps.f16 f1, f1\n"
                       "fadd.ps f0, f0, f1 \n"
                       "2:\n"

                       "addi t0, t0, 0x1\n"
                       "addi t3, t3, 0x8\n"
                       "ble t0, %[numIndices], 1b\n"

                       "fcvt.f16.ps f0, f0\n"
                       "fsch.ps  f0, f31(%[dst]) \n"

                       :
                       : [ gatherValues ] "m" (* ( const int32_t(*)[8]) gatherValues),
                         [ src ] "r"(src),
                         [ numIndices ] "r"(numIndices),
                         [ batch ] "r"(batch),
                         [ tIndex ] "r"(tIndex),
                         [ batchPitch ] "r"(batchPitch),
                         [ typeSize ] "r"(typeSize),
                         [ dst ] "r"(dst)
                       : "t0", "t1", "t2", "t3", "f0", "f1", "f31", "memory");

}



template <ElemKind elK, typename std::enable_if< elK == Int8QTy,std::size_t>::type = 0>
inline __attribute__((always_inline)) void sparseToDenseOp (uintptr_t dst, uintptr_t src, long long* tIndex, unsigned int batchPitch,
unsigned int batch, unsigned int numIndices, size_t typeSize, const float *scale, const int32_t *offset){
  int32_t gatherValues[] = {0, 1, 2, 3, 4, 5, 6, 7};
  __asm__ __volatile__("add t0, zero, zero\n"
                       "fxor.pi f0, f0, f0\n"
                       "flw.ps f31, %[gatherValues]\n"
                       "fbc.ps f30, 0x0(%[offset]) \n"
                       "fbc.ps f29, 0x0(%[scale]) \n"

                       "fsub.pi f0, f0, f30 \n"
                       "fcvt.ps.pw f0, f0 \n"
                       "fmul.ps f0, f0, f29 \n"


                       "addi    t3, %[tIndex], 0x0\n"
                       "1:\n"

                       "ld t1, 0x0(t3)\n"
                       "bne t1, %[batch], 2f\n"

                       "mul t2, t0, %[typeSize]\n"
                       "mul t2, t2, %[batchPitch]\n"
                       "add t2, t2, %[src]\n"

                       "fgb.ps  f1, f31(t2) \n"
                       "fsub.pi f1, f1, f30 \n"
                       "fcvt.ps.pw f1, f1 \n"

                       "fmadd.ps f0, f1, f29, f0 \n"

                       "2:\n"

                       "addi t0, t0, 0x1\n"
                       "addi t3, t3, 0x8\n"
                       "ble t0, %[numIndices], 1b\n"

                       "frcp.ps f29, f29 \n"
                       "fcvt.ps.pw f30, f30 \n"
                       "fmadd.ps f0, f0, f29, f30 \n"
                       "fcvt.pw.ps f0, f0 \n"
                       "fsat8.pi f0, f0 \n"
                       "fscb.ps  f0, f31(%[dst]) \n"



                       :
                       : [ gatherValues ] "m" (* ( const int32_t(*)[8]) gatherValues),
                         [ src ] "r"(src),
                         [ numIndices ] "r"(numIndices),
                         [ batch ] "r"(batch),
                         [ tIndex ] "r"(tIndex),
                         [ batchPitch ] "r"(batchPitch),
                         [ typeSize ] "r"(typeSize),
                         [ dst ] "r"(dst),
                         [ offset ] "r"(offset),
                         [ scale ] "r"(scale)

                       : "t0", "t1", "t2", "t3", "f0", "f1", "f29", "f30", "f31", "memory");

}


template <ElemKind elK, typename std::enable_if<elK!=Int8QTy && elK!=Float16Ty && elK!=FloatTy, std::size_t>::type = 0>
inline __attribute__((always_inline)) void sparseToDenseOp (uintptr_t dst, uintptr_t src, long long* tIndex, unsigned int batchPitch,
unsigned int batch, unsigned int numIndices, size_t typeSize, const float *scale, const int32_t *offset){
}

  
  // vetorized version
template <ElemKind elK>
inline __attribute__((always_inline)) void fwdLibSparseToDenseInst(
                                                                   LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                                                   uint64_t flags,
                                                                   const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  float scale[] = { in2T->getScale(), in1T->getScale(), outT->getScale()};
  int32_t offset[] = { in2T->getOffset(), in1T->getOffset(), outT->getOffset()};
  /* outT --> dst  in2T--> src   in1T--> indices */
  /* maintain compatibility through the new Iface Libtensor */

  void* dstT = outT->getRawDataPointer<void>();
  void* srcT = in2T->getRawDataPointer<void>();
  // long long *tIndex = (long long *)indicesT;
  long long *tIndex = in1T->getRawDataPointer<long long>();
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *indIndex = (unsigned int *)indDims;
  const dim_t *indIndex = in1T->dims().data();
  
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = in2T->strides().data();
  
  unsigned int srcDimNum = static_cast<unsigned int>(in2T->ndims());
  
  uintptr_t dstAddr = (uintptr_t)dstT;    
  uintptr_t srcAddr = (uintptr_t)srcT;


  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = sizeof(srcType);
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    coord[i] = 0;
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;

  unsigned int batchPitch = srcPitch[0];
  // @TODO srcpitch It is a cnst pointer!!!!. Re-do in other way
  // it is not allowed modify tensor properties. It needs a cpy of it.
  size_t cpySrcPitch[srcDimNum] = {0,};
  for (size_t i = 0; i < srcDimNum; i++)
    cpySrcPitch[i] = srcPitch[i];
  //srcPitch[0] = 0;
  cpySrcPitch[0] = 0;

  
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
    offsetIn += cpySrcPitch[j] * coord[j];
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


    for (unsigned int i = 0; i < registersInRow; i++) {
      sparseToDenseOp <elK>(dstAddr, srcAddr, tIndex, batchPitch, coord[0], indIndex[0], typeSize, scale, offset);
      srcAddr += 8 * typeSize;
      dstAddr += 8 * typeSize;
    }
    if(res > 0) {
      __asm__ __volatile__("maskand m0, m1, m0 \n");
      sparseToDenseOp <elK>(dstAddr, srcAddr, tIndex, batchPitch, coord[0], indIndex[0], typeSize, scale, offset);
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
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _SPARSE_TO_DENSE_INST_H_
