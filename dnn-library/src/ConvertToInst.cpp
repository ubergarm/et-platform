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

template <typename srcType, typename dstType>
void dnn_lib::fwdLibConvertToInst(void *dstT, void *dstDims, void *dstPitches,
                                  void *srcT1, void *srcDims, void *srcPitches,
                                  unsigned int srcDimNum, float *scale,
                                  int32_t *offset) {

  // FIXME: single thread convertto fails when combined with multi-threaded
  // operators
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<dstType> ptrDstT(dstT, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT1(srcT1, scale[0], offset[0]);

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
  Converter<srcType, dstType> converter;
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
              auto src = ptrSrcT1[addrSrc];
              if (std::is_same<srcType, dstType>::value) {
                ptrDstT[addrDst] = src;
              } else {
                auto dst = converter.convert(src);
                ptrDstT[addrDst] = dst;
              }
            }
          }
        }
      }
    }
  }
}

template <typename srcType, typename dstType>
void dnn_lib::fwdLibConvertToInstThreaded(void *dst, void *dstDims,
                                          void *dstPitches, void *src,
                                          void *srcDims, void *srcPitches,
                                          unsigned int srcDimNum, float *scale,
                                          int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<dstType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  Converter<srcType, dstType> converter;

  unsigned int numElemsDst =
      dstPitch[0] * actIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
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
    auto input = tAInput[offsetIn];
    auto output = converter.convert(input);
    tOutput[offsetOut] = output;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


template <typename srcType, typename dstType>
void dnn_lib::fwdLibConvertToInstVectorized(void *dst, void *dstDims,
                                            void *dstPitches, void *src,
                                            void *srcDims, void *srcPitches,
                                            unsigned int srcDimNum, float *scale,
                                            int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  //Addresser<dstType> tOutput(dst, scale[1], offset[1]);
  //const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  uintptr_t srcAddr = (uintptr_t)src;
  uintptr_t dstAddr = (uintptr_t)dst;

  Converter<srcType, dstType> converter;

  unsigned int numElemsDst =
      dstPitch[0] * dstIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSizeSrc = getsize<srcType>();
  size_t typeSizeDst = getsize<dstType>();
  getCachelinePartition(typeSizeDst, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (unlikely(maxRead == 0))
    return;

  // We move the initialAddr to the next non-padding position
  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k;                  // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex,
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

  unsigned int lastDim = srcDimNum - 1;

  volatile int32_t gatherValues[] = {0, 0, 0, 0, 0, 0, 0, 0};
  volatile int32_t scatterValues[] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (unsigned int i = 0; i < 8; i++) {
      gatherValues[i] = i * typeSizeSrc;
      scatterValues[i] = i * typeSizeDst;
  }

  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  unsigned int elementsInRow, registersInRow, res, spareElems, fullLanes;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  unsigned int elementsInRegister =  8 * (typeSizeDst != 8) + 4 * (typeSizeDst == 8);
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
      registersInRow = elementsInRow / elementsInRegister;
      res = elementsInRow - registersInRow * elementsInRegister;
      if (elementsInRegister != 4) {
        mask = ((1 << res) - 1);
      } else {
        mask = ((1 << 2*res) - 1);
      }
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr += offsetIn * typeSizeSrc;
    dstAddr += offsetOut * typeSizeDst;
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

    unsigned int cnt = 0;
    while(cnt < registersInRow) {
      converter.convertVect(srcAddr, dstAddr, gatherValues, scatterValues);
      cnt++;
      srcAddr += typeSizeSrc * elementsInRegister;
      dstAddr += typeSizeDst * elementsInRegister;
    }
    if (res > 0) {
      if (elementsInRegister != 4) {
        __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      } else {
        __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n"
                             "mov.m.x    m1, zero, 0x55 \n"
                             "maskand m0, m0, m1 \n"
                             : : [ mask ] "r"(mask) :);
      }
      converter.convertVect(srcAddr, dstAddr, gatherValues, scatterValues);
    }


    if (lastRow)
      return;

    srcAddr = (uintptr_t)src;
    dstAddr = (uintptr_t)dst;
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;
    done = getOffsets(lastDim , coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSizeDst / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSizeDst*initialAddr, clperminion);
}

GEN_INSTANCES_CONVERT(template, fwdLibConvertToInst, void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT1, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, float *scale, int32_t *offset);
GEN_INSTANCES_CONVERT(template, fwdLibConvertToInstThreaded, void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT1, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_CONVERT(template, fwdLibConvertToInstVectorized, void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT1, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, float *scale, int32_t *offset, uint64_t flags);
