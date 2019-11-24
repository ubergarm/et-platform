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

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "LibNodes.h"
#include "GenInstances.h"
#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"

using namespace std;

// This function copies a matrix replacing all the elements which are < splatVal
// and replaces them with splatVal
template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *srcT,
                                      void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, float splatVal,
                                      float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT(srcT, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              auto src = ptrSrcT[addrSrc];
              ptrDstT[addrDst] = (src > splatVal) ? src : splatVal;
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *srcT,
                                      void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, int64_t splatVal,
                                      float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  srcType *ptrDstT = (srcType *)dstT;
  srcType *ptrSrcT = (srcType *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              int64_t src = ptrSrcT[addrSrc];
              ptrDstT[addrDst] = (src > splatVal) ? src : splatVal;
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInstThreaded(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              float splatVal, float *scale,
                                              int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> ptrDstT(dst, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

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
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInstThreaded(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              int64_t splatVal, float *scale,
                                              int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  srcType *ptrDstT = (srcType *)dst;
  srcType *ptrSrcT = (srcType *)src;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

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
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value,
std::size_t>::type = 0>
void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset){
  volatile int32_t gatherValues[] = {0, 4, 8, 12, 16, 20, 24, 28};
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n"
                       "fgw.ps f0, f31(%[src])\n"
                       "fbc.ps f1, 0x0(%[splatVal])\n"
                       "fmax.ps f0, f0, f1\n"
                       "fscw.ps  f0, f31(%[dst]) \n"

                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ splatVal ] "r"(&splatVal),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "f0", "f1", "f31", "memory");



}


template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value,
std::size_t>::type = 0>
void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset){
  volatile int32_t gatherValues[] = {0, 2, 4, 6, 8, 10, 12, 14};
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n"
                       "fgh.ps f0, f31(%[src])\n"
                       "fcvt.ps.f16 f0, f0\n"
                       "fbc.ps f1, 0x0(%[splatVal])\n"
                       "fmax.ps f0, f0, f1\n"
                       "fcvt.f16.ps f0, f0\n"
                       "fsch.ps  f0, f31(%[dst]) \n"

                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ splatVal ] "r"(&splatVal),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "f0", "f1", "f31", "memory");
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value,
std::size_t>::type = 0>
void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset ){

  volatile int32_t gatherValues[] = {0, 1, 2, 3, 4, 5, 6, 7};
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n"
                       "fgb.ps f0, f31(%[src])\n"
                       "fbc.ps f30, 0x0(%[offset]) \n"
                       "fbc.ps f29, 0x0(%[scale]) \n"
                       "fsub.pi f0, f0, f30 \n"
                       "fcvt.ps.pw f0, f0 \n"
                       "fmul.ps f0, f0, f29 \n"
                       "fbc.ps f1, 0x0(%[splatVal])\n"
                       "fmax.ps f0, f0, f1\n"
                       "frcp.ps f29, f29 \n"
                       "fcvt.ps.pw f30, f30 \n"
                       "fmadd.ps f0, f0, f29, f30 \n"
                       "fcvt.pw.ps f0, f0 \n"
                       "fsat8.pi f0, f0 \n"
                       "fscb.ps  f0, f31(%[dst]) \n"
                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ splatVal ] "r"(&splatVal),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src),
                         [ offset ] "r"(offset),
                         [ scale ] "r"(scale)

                       : "f0", "f1", "f29", "f30", "f31", "memory");
}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value
&& !std::is_same<srcType, float16>::value
&& !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset ){}

template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInstVectorized(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              float splatVal, float *scale,
                                              int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  uintptr_t dstAddr = (uintptr_t)dst;
  uintptr_t srcAddr = (uintptr_t)src;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

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
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value,
std::size_t>::type = 0>
void maxSplatOpAligned32Bytes (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset){
  __asm__ __volatile__("flw.ps f0, 0x0(%[src])\n"
                       "fbc.ps f1, 0x0(%[splatVal])\n"
                       "fmax.ps f0, f0, f1\n"
                       "fsw.ps  f0, 0x0(%[dst]) \n"
                       :
                       : [ splatVal ] "r"(&splatVal),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "f0", "f1", "f31", "memory");



}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value,
std::size_t>::type = 0>
void maxSplatOpAligned32Bytes (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset){
  __asm__ __volatile__( SET_FG32H_VAL(t0)
                       "fg32h.ps f0, t0(%[src])\n"
                       "fcvt.ps.f16 f0, f0\n"
                       "fbc.ps f1, 0x0(%[splatVal])\n"
                       "fmax.ps f0, f0, f1\n"
                       "fcvt.f16.ps f0, f0\n"
                       "fsc32h.ps f0, t0(%[dst]) \n"

                       :
                       : [ splatVal ] "r"(&splatVal),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "t0", "f0", "f1", "f31", "memory");
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value,
std::size_t>::type = 0>
void maxSplatOpAligned32Bytes (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset ){

  __asm__ __volatile__(SET_FG32B_VAL(t0)
                       "fg32b.ps f0, t0(%[src])\n"
                       "fbc.ps f30, 0x0(%[offset]) \n"
                       "fbc.ps f29, 0x0(%[scale]) \n"
                       "fsub.pi f0, f0, f30 \n"
                       "fcvt.ps.pw f0, f0 \n"
                       "fmul.ps f0, f0, f29 \n"
                       "fbc.ps f1, 0x0(%[splatVal])\n"
                       "fmax.ps f0, f0, f1\n"
                       "frcp.ps f29, f29 \n"
                       "fcvt.ps.pw f30, f30 \n"
                       "fmadd.ps f0, f0, f29, f30 \n"
                       "fcvt.pw.ps f0, f0 \n"
                       "fsat8.pi f0, f0 \n"
                       "fsc32b.ps f0, t0(%[dst]) \n"
                       :
                       : [ splatVal ] "r"(&splatVal),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src),
                         [ offset ] "r"(offset),
                         [ scale ] "r"(scale)

                       : "t0", "f0", "f1", "f29", "f30", "f31", "memory");
}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value
&& !std::is_same<srcType, float16>::value
&& !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void maxSplatOpAligned32Bytes (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset ){}

template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInstAligned32Bytes(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              float splatVal, float *scale,
                                              int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  uintptr_t dstAddr;
  uintptr_t srcAddr;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

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
  unsigned int res = ((dstIndex[lastDim] - 1)%8) +1;
  actPitch[lastDim] *= 8;
  dstPitch[lastDim] *= 8;
  dstIndex[lastDim] = (dstIndex[lastDim] - 1)/8 + 1;
  unsigned int mask = ((1 << res) - 1);

  while (!done && (offsetOut < posMax)) {
    dstAddr = (uintptr_t)dst + offsetOut*typeSize;
    srcAddr = (uintptr_t)src + offsetIn*typeSize;

    if (coord[lastDim] != dstIndex[lastDim] - 1)
         __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
    else __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);

    maxSplatOpAligned32Bytes <srcType>(dstAddr, srcAddr, splatVal, scale, offset);

    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0)
    evict_va(0, DO_EVICTS, initialAddr, clperminion - 1, 64);
}

GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInst, void *dstT, void *dstDims, void *dstPitches,
                                void *srcT, void *srcDims, void *srcPitches,
                                unsigned int srcDimNum, float splatVal,
                                float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInst, void *dstT, void *dstDims, void *dstPitches,
                                void *srcT, void *srcDims, void *srcPitches,
                                unsigned int srcDimNum, int64_t splatVal,
                                float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInstThreaded,void *dst, void *dstDims, void *dstPitches,
                                         void *src, void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float splatVal,
                                         float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInstThreaded,void *dst, void *dstDims, void *dstPitches,
                                         void *src, void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, int64_t splatVal,
                                         float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInstVectorized,void *dst, void *dstDims, void *dstPitches,
                                         void *src, void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float splatVal,
                                         float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInstAligned32Bytes,void *dst, void *dstDims, void *dstPitches,
                                         void *src, void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float splatVal,
                                         float *scale, int32_t *offset, uint64_t flags);
