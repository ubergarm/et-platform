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

template <ElemKind elK>
inline __attribute__((always_inline)) void fwdLibSparseToDenseInst(LibTensor* outT,
                                                                   LibTensor* in1T,
                                                                   LibTensor* in2T,
                                                                   uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  //  using srcType = typename elemKind2elemTy<elK>::type;

  if (get_minion_id() != minionOffset) return;
  
  /* outT --> dst  in1T--> src   in2T--> indices */
  /* maintain compatibility through the new Iface Libtensor */

  void* dstT = outT->getRawDataPointer<void>();
  void* srcT = in1T->getRawDataPointer<void>();

  // const Addresser<elK> tInput(srcT, scale[0], offset[0]);
  const Addresser<elK> tInput(srcT, in1T->getScale(), in1T->getOffset());
  // Addresser<elK> tOutput(dstT, scale[2], offset[2]);
  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tTmp(dstT, scale[2], offset[2]);
  const Addresser<elK> tTmp(dstT, outT->getScale(), outT->getOffset());
  // long long *tIndex = (long long *)indicesT;
  long long *tIndex = in2T->getRawDataPointer<long long>();
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *indIndex = (unsigned int *)indDims;
  const dim_t *indIndex = in2T->dims().data();
  
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = in1T->strides().data();

  unsigned int srcDimNum = static_cast<unsigned int>(in1T->ndims());
  
  // Convert sparse representation to dense representation by taking
  // slices of output and values and accumulating the value slice into
  // the output slice.

  // Dimensions and coord for the output and values slices. sliceDims
  // will always be {1, [rest of output dimensions]} since the first dimension
  // is the index in this operation. sliceOffsets will be {indices[j], 0, ...}
  // for the output slice and {j, 0, ...} for the values slice so that the
  // slice at index j gets mapped to index indices[j] in the dense
  // representation.

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = dstIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }
  eBatchDims[0] = 1;
  uint64_t addrSrc, addrDst;

  // Initialize to zero output tensor
  for (size_t j = 0; j < dstIndex[0]; j++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrDst = j * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] 
                      + w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              tOutput[addrDst] = 0;
            }
          }
        }
      }
    }
  }

  for (size_t j = 0; j < indIndex[0]; j++) {
    long long index = tIndex[j];
    // We can use this loop for all shapes.
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc = j * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = index * eDstPitch[0] + y * eDstPitch[1] +
                        z * eDstPitch[2] + w * eDstPitch[3] + q * eDstPitch[4] +
                        r * eDstPitch[5];
              tOutput[addrDst] = tTmp[addrDst] + tInput[addrSrc];
            }
          }
        }
      }
    }
  }
}

template <ElemKind elK>
inline __attribute__((always_inline)) void fwdLibSparseToDenseInstThreaded(
            LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,
            const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* outT --> dst  in1T--> src   in2T--> indices */
  /* maintain compatibility through the new Iface Libtensor */

  void* dstT = outT->getRawDataPointer<void>();
  void* srcT = in1T->getRawDataPointer<void>();

  // const Addresser<elK> tInput(srcT, scale[0], offset[0]);
  const Addresser<elK> tInput(srcT, in1T->getScale(), in1T->getOffset());
  // Addresser<elK> tOutput(dstT, scale[2], offset[2]);
  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  // long long *tIndex = (long long *)indicesT;
  long long *tIndex = in2T->getRawDataPointer<long long>();
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *indIndex = (unsigned int *)indDims;
  const dim_t *indIndex = in2T->dims().data();  

  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = in1T->strides().data();

  unsigned int srcDimNum = static_cast<unsigned int>(in1T->ndims());
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    coord[i] = 0;
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetIn = 0; // Doesn't include srcPitch[0]. offsetIn doesn't
                             // have the conventional meaning
  unsigned int offsetOut = 0;

  unsigned int srcPitch_0 = srcPitch[0];
  // @TODO srcpidtch It is a cnst pointer!!!!. Re-do in other way
  // it is not allowed modify tensor properties. It needs a cpy of it.
  size_t cpySrcPitch[srcDimNum] = {0,};
  for (size_t i = 0; i < srcDimNum; i++)
    cpySrcPitch[i] = srcPitch[i];
  // srcPitch[0] = 0;
  cpySrcPitch[0] = 0;
  
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
    offsetIn += cpySrcPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    srcType sum;
    sum = 0;
    for (size_t j = 0; j < indIndex[0]; j++) {
      if (tIndex[j] == coord[0]) sum = sum + tInput[offsetIn + j * srcPitch_0];
    }
    tOutput[offsetOut] = sum;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, dstIndex,
                      cpySrcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value,
std::size_t>::type = 0>
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

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value,
std::size_t>::type = 0>
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



template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value,
std::size_t>::type = 0>
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


template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value
&& !std::is_same<srcType, float16>::value
&& !std::is_same<srcType, float>::value, std::size_t>::type = 0>
inline __attribute__((always_inline)) void sparseToDenseOp (uintptr_t dst, uintptr_t src, long long* tIndex, unsigned int batchPitch,
unsigned int batch, unsigned int numIndices, size_t typeSize, const float *scale, const int32_t *offset){
}


template <ElemKind elK>
inline __attribute__((always_inline)) void fwdLibSparseToDenseInstVectorized(
           LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
           uint64_t flags,
           const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  float scale[] = { in1T->getScale(), in2T->getScale(), outT->getScale()};
  int32_t offset[] = { in1T->getOffset(), in2T->getOffset(), outT->getOffset()};
  /* outT --> dst  in1T--> src   in2T--> indices */
  /* maintain compatibility through the new Iface Libtensor */

  void* dstT = outT->getRawDataPointer<void>();
  void* srcT = in1T->getRawDataPointer<void>();
  // long long *tIndex = (long long *)indicesT;
  long long *tIndex = in2T->getRawDataPointer<long long>();
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *indIndex = (unsigned int *)indDims;
  const dim_t *indIndex = in2T->dims().data();
  
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = in1T->strides().data();
  
  unsigned int srcDimNum = static_cast<unsigned int>(in1T->ndims());
  
  uintptr_t dstAddr = (uintptr_t)dstT;    
  uintptr_t srcAddr = (uintptr_t)srcT;


  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
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
      sparseToDenseOp <srcType>(dstAddr, srcAddr, tIndex, batchPitch, coord[0], indIndex[0], typeSize, scale, offset);
      srcAddr += 8 * typeSize;
      dstAddr += 8 * typeSize;
    }
    if(res > 0) {
      __asm__ __volatile__("maskand m0, m1, m0 \n");
      sparseToDenseOp <srcType>(dstAddr, srcAddr, tIndex, batchPitch, coord[0], indIndex[0], typeSize, scale, offset);
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
