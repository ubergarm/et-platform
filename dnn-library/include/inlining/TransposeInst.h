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

namespace dnn_lib {

namespace inlining {

template <typename srcType>
inline void fwdLibTransposeInst(void *dst, void *dstDims, void *dstPitches,
                                  void *src, void *srcDims, void *srcPitches,
                                  unsigned int srcDimNum, void *pshuffle,
                                  const float *scale, const int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int *shuffle = (unsigned int *)pshuffle;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  // Iterates through all dimensions, and sets extended Dims, and src Pitches
  for (int i = 0; i < srcDimNum; i++) {
    // extended Dims matches src dim and zeros for non used dims
    eDims[i] = actIndex[i];
    // extended src Pitches matches src Pitches and zeros for non used dims
    eSrcPitch[i] = actPitch[i];
    for (int j = 0; j < srcDimNum; j++) {
      if (shuffle[j] == i) {
        eDstPitch[i] = dstPitch[j];
      }
    }
  }

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              uint64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                 z * eDstPitch[2] + w * eDstPitch[3] +
                                 q * eDstPitch[4] + r * eDstPitch[5];
              uint64_t srcAddr = x * eSrcPitch[0] + y * eSrcPitch[1] +
                                 z * eSrcPitch[2] + w * eSrcPitch[3] +
                                 q * eSrcPitch[4] + r * eSrcPitch[5];
              tOutput[dstAddr] = tAInput[srcAddr];
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
inline void fwdLibTransposeInstThreaded(void *dst, void *dstDims,
                                          void *dstPitches, void *src,
                                          void *srcDims, void *srcPitches,
                                          unsigned int srcDimNum,
                                          void *pshuffle, const float *scale,
                                          const int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int *shuffle = (unsigned int *)pshuffle;

  uintptr_t dstAddr = (uintptr_t)dst;
  uintptr_t srcAddr = (uintptr_t)src;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int newPitch[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    newPitch[i] = actPitch[shuffle[i]];

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
  while (not done && offsetOut < posMax) {
    tOutput[offsetOut] = tAInput[offsetIn];
    done = getOffsets(srcDimNum, coord, offsetOut, offsetIn, dstIndex, dstPitch, newPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <typename srcType, typename std::enable_if< (getsize<srcType>() <=4), int>::type = 0>
void transposeOp (uintptr_t dst, uintptr_t src, int32_t *scatterValues, int32_t *gatherValues){
  constexpr size_t size = getsize<srcType>();
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
                         [ size] "i" (size)
                       : "f0", "f31", "memory");
}

  template <typename srcType, typename std::enable_if< (getsize<srcType>()>4), int>::type = 0 >
void transposeOp (uintptr_t dst, uintptr_t src, int32_t *scatterValues,  int32_t *gatherValues){
  //FIXME: TODO: implement
}



template <typename srcType>
inline void fwdLibTransposeInstVectorized(void *dst, void *dstDims,
                                            void *dstPitches, void *src,
                                            void *srcDims, void *srcPitches,
                                            unsigned int srcDimNum,
                                            void *pshuffle, const float *scale,
                                            const int32_t *offset, uint64_t flags) {
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

  unsigned int *shuffle = (unsigned int *)pshuffle;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int newPitch[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    newPitch[i] = actPitch[shuffle[i]];

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
  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;


  unsigned int elementsInRow, registersInRow, res;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);


  unsigned int newPitchSize = newPitch[lastDim] * typeSize;
  int32_t gatherValues[8];
  for (unsigned int i = 0; i < 8; i++) gatherValues[i] = i*newPitchSize;
  unsigned int dstPitchSize = dstPitch[lastDim] * typeSize;
  int32_t scatterValues[8];
  for (unsigned int i = 0; i < 8; i++) scatterValues[i] = i*dstPitchSize;

  while (!done && (offsetOut < posMax)) {
    if (firstRow && (coord[lastDim - 1] != maxRow)) {
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
      transposeOp <srcType>(dstAddr, srcAddr, scatterValues, gatherValues);
      srcAddr += 8 * typeSize * newPitch[lastDim];
      dstAddr += 8 * typeSize;
    }

    if (res > 0) {
      __asm__ __volatile__("maskand m0, m1, m0 \n");
      transposeOp <srcType>(dstAddr, srcAddr, scatterValues, gatherValues);
    }

    if (lastRow)
      return;

    dstAddr = (uintptr_t)dst;
    srcAddr = (uintptr_t)src;
    offsetIn -= coord[lastDim] * newPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;

    done = getOffsets(lastDim, coord, offsetOut, offsetIn, dstIndex, dstPitch, newPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


template <typename srcType, typename std::enable_if< (getsize<srcType>() <=4), int>::type = 0>
void transposeOpAligned32Bytes (uintptr_t dst, uintptr_t src, int32_t *gatherValues){
  constexpr size_t size = getsize<srcType>();
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
  
  template <typename srcType, typename std::enable_if< (getsize<srcType>() >4), int>::type = 0>
  void transposeOpAligned32Bytes (uintptr_t dst, uintptr_t src, int32_t *gatherValues){
  //FIXME: not implemented
}



template <typename srcType>
inline void fwdLibTransposeInstAligned32Bytes(void *dst,
                                          void *dstDims,
                                          void *dstPitches, void *src,
                                          void *srcDims, void *srcPitches,
                                          unsigned int srcDimNum,
                                          void *pshuffle, const float *scale,
                                          const int32_t *offset, uint64_t flags) {
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

  unsigned int *shuffle = (unsigned int *)pshuffle;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int newPitch[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    newPitch[i] = actPitch[shuffle[i]];

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
  dstPitch[lastDim] *= 8;
  dstIndex[lastDim] = (dstIndex[lastDim] - 1)/8 + 1;
  unsigned int mask = ((1 << res) - 1);

  while (!done && (offsetOut < posMax)) {
    dstAddr = (uintptr_t)dst + offsetOut*typeSize;
    srcAddr = (uintptr_t)src + offsetIn*typeSize;

    //When the minion reaches the end of the lastDim, we use a mask
    //that is always the same because the dst Tensor is aligned to 32 Bytes.
    if (coord[lastDim] != dstIndex[lastDim] - 1)
         __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
    else __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);

    transposeOpAligned32Bytes <srcType>(dstAddr, srcAddr, gatherValues);
    done = getOffsets(srcDimNum, coord, offsetOut, offsetIn, dstIndex, dstPitch, newPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0)
    evict_va(0, DO_EVICTS, initialAddr, clperminion - 1, 64);
}

} // namespace inlining

} // namespace dnn_lib

#endif //  _TRANSPOSE_INST_H_
