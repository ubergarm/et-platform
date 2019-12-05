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

template <typename srcType>
void dnn_lib::fwdLibCopyInst(void *dst, void *dstDims, void *dstPitches,
                             void *src, void *srcDims, void *srcPitches,
                             unsigned int srcDimNum, float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tInput(src, scale[0], offset[0]);

  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;
  
  unsigned int coord[srcDimNum] = {0};
  unsigned int offsetIn  = 0;
  unsigned int offsetOut = 0;

  bool done = false;

  while (!done) {
    tOutput[offsetOut] = tInput[offsetIn];
    done = getNextStep(srcDimNum, coord, actIndex);
    offsetIn  = getOffset(coord, srcDimNum, actPitch);
    offsetOut = getOffset(coord, srcDimNum, dstPitch);
  }
}


/* This is currently the fastest version of the copy, geting up to 800x velocity
 * compared to the single thread version of CopyInst. It splits the total
 * cachelines in packs and distributes them between all the minions possible.
 *
 * It is specially more desirable than other threading implementations
 * due to the fact that it updates the position being read and written via
 * the sum of the pitch, instead of computing it on each iteration of the loop.
 *
 * Moreover, this version gets over the limited convention of only considering
 * arrays of 6 dimensions, as extended vectors are not needed, and therefore
 * this implementation is a generalization of the previous ones. */

template <typename srcType>
void dnn_lib::fwdLibCopyInstThreaded(void *dst, void *dstDims,
                                     void *dstPitches, void *src,
                                     void *srcDims, void *srcPitches,
                                     unsigned int srcDimNum, float *scale,
                                     int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  uint8_t *dst8 = (uint8_t *)dst;
  uint8_t *src8 = (uint8_t *)src;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst =
      dstPitch[0] * dstIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address and the number of positions that
  // it must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position

  unsigned int k;                  // Amount of non-zero coordinates
  unsigned int coord[srcDimNum]; // Vector of coordinates

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
    tOutput[offsetOut] = tAInput[offsetIn];
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


// Vectorized and threaded general version of CopyInst. Work in progress.

template <typename srcType>
void dnn_lib::fwdLibCopyInstVectorized(void *dst, void *dstDims,
                                       void *dstPitches, void *src,
                                       void *srcDims, void *srcPitches,
                                       unsigned int srcDimNum, float *scale,
                                       int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;
  uint8_t *dst8 = (uint8_t *)dst;
  uint8_t *src8 = (uint8_t *)src;
  uint8_t *src8Init = src8;
  uint8_t *dst8Init = dst8;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int typeSize = getsize<srcType>();
  unsigned int numElemsDst =
      dstPitch[0] * actIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;
  // We move the initialAddr to the next non-padding position
  unsigned int k;                  // Amount of non-zero coordinates
  unsigned int coord[srcDimNum]; // Vector of coordinates
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
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;

  unsigned int laneElems = 4 / typeSize;
  unsigned int registerElems;
  if (laneElems != 0) {
    registerElems = 8 * laneElems;
  } else {
    registerElems = 4;
  }
  unsigned int maxRow = (srcDimNum > 1) ? posMax / dstPitch[lastDim - 1] : 0;
  unsigned int elementsInRow, registersInRow, res, spareElems, fullLanes;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

  while ((offsetOut < posMax) && !done) {
    if (lastDim != 0) {
      if (firstRow && (coord[lastDim - 1] != maxRow)) {
        elementsInRow = dstIndex[lastDim] - coord[lastDim];
      } else if (coord[lastDim - 1] == maxRow) {
        lastRow = true;
        elementsInRow = posMax - offsetOut;
      }
      else {
      elementsInRow = dstIndex[lastDim];
      }
    } else {
      elementsInRow = dstIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / registerElems;
      res = elementsInRow - registersInRow * registerElems;
      if (laneElems != 0) {
        fullLanes = res / laneElems;
        spareElems = res - fullLanes * laneElems;
      } else {
        fullLanes = res * 2;
        spareElems = 0;
      }
      mask = ((1 << fullLanes) - 1);
      __asm__ __volatile__("mov.m.x m1, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    src8 += offsetIn * typeSize;
    dst8 += offsetOut * typeSize;

    unsigned int cnt = 0;
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
    while (cnt < registersInRow) {
      __asm__ __volatile__("flw.ps f0, 0x0(%[src]) \n"
                           "fsw.ps f0, 0x0(%[dst]) \n"
                           :
                           : [ src ] "r"(src8), [ dst ] "r"(dst8)
                           : "f0", "memory");
      src8 += 32;
      dst8 += 32;
      cnt++;
    }
    __asm__ __volatile__("maskand m0, m0, m1 \n"
                         "flw.ps f0, 0x0(%[src]) \n"
                         "fsw.ps f0, 0x0(%[dst]) \n"
                         "mov.m.x m0, zero, 0xff \n"
                         :
                         : [ src ] "r"(src8), [ dst ] "r"(dst8)
                         : "f0", "memory");
    src8 += fullLanes * 4;
    dst8 += fullLanes * 4;
    unsigned int offsetInAux = (src8 - src8Init) / typeSize;
    unsigned int offsetOutAux = (dst8 - dst8Init) / typeSize;
    for (unsigned int i = 0; i < spareElems; i++) {
      tOutput[offsetOutAux + i] = tAInput[offsetInAux + i];
    }

    if (lastRow)
      return;
    src8 = src8Init;
    dst8 = dst8Init;
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;

    done = getOffsets(srcDimNum - 1, coord, offsetIn, offsetOut, actIndex, actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

// template <typename srcType>
// void dnn_lib::fwdLibCopyInstVectorizedGeneral(void *dst, void *dstDims,
//                                        void *dstPitches, void *src,
//                                        void *srcDims, void *srcPitches,
//                                        unsigned int srcDimNum, float *scale,
//                                        int32_t *offset, uint64_t flags) {
//   Addresser<srcType> tOutput(dst, scale[1], offset[1]);
//   const Addresser<srcType> tAInput(src, scale[0], offset[0]);
//
//   unsigned int *dstIndex = (unsigned int *)dstDims;
//   unsigned int *actIndex = (unsigned int *)srcDims;
//   int8_t *dst8 = (int8_t *)dst;
//   int8_t *src8 = (int8_t *)src;
//
//   unsigned int *dstPitch = (unsigned int *)dstPitches;
//   unsigned int *actPitch = (unsigned int *)srcPitches;
//
//   unsigned int minionId = get_minion_id();
//   unsigned int activeMinions = 32 * ACTIVE_SHIRES;
//   if (minionId >= activeMinions) {
//     return;
//   }
//   unsigned int typeSize = getsize<srcType>();
//   unsigned int numElemsDst =
//       dstPitch[0] * actIndex[0]; // Total number of elements in the tensor
//
//   // We give to each minion an initial address the number of positions that it
//   // must work on (maxRead).
//   unsigned int initialAddr, maxRead;
//   getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
//                         activeMinions);
//   if (maxRead == 0)
//     return;
//   // We move the initialAddr to the next non-padding position
//   unsigned int k;                  // Amount of non-zero coordinates
//   unsigned int coord[srcDimNum]; // Vector of coordinates
//   getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
//                            k);
//
//   // We get the actual initialAddr, in the input and output.
//   unsigned int offsetIn = 0;
//   unsigned int offsetOut = 0;
//   for (unsigned int j = 0; j < k; j++) {
//     offsetIn += actPitch[j] * coord[j];
//     offsetOut += dstPitch[j] * coord[j];
//   }
//
//   unsigned int posMax = maxRead + initialAddr;
//   bool done = false;
//   unsigned int lastDim = srcDimNum - 1;
//   if (actIndex[lastDim] < 4) {
//     while ((offsetOut < posMax) && (not done)) {
//       tOutput[offsetOut] = tAInput[offsetIn];
//       done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
//                         actPitch, dstPitch);
//     }
//   } else {
//     while ((coord[lastDim] != 0) && (offsetOut < posMax) && (not done)) {
//       tOutput[offsetOut] = tAInput[offsetIn];
//       done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
//                         actPitch, dstPitch);
//     }
//     unsigned int registerElems = 32 / typeSize;
//     unsigned int res = dstIndex[lastDim] % registerElems;
//     unsigned int maxAux = posMax - res;
//     unsigned int limit = dstIndex[lastDim] - res;
//     src8 += offsetIn * typeSize;
//     dst8 += offsetOut * typeSize;
//     __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
//     __asm__ __volatile__("mov.m.x m1, zero, 0xff \n");
//     uint8_t mask = (uint8_t)((1 << res) - 1);
//     __asm__ __volatile__("mov.m.x m2, %[mask], 0 \n" : : [ mask ] "r"(&mask) :);
//     while ((offsetOut < maxAux) && (not done)) {
//       if (coord[lastDim] < limit) {
//         __asm__ __volatile__("flw.ps f0, 0x0(%[src]) \n"
//                              "fsw.ps f0, 0x0(%[dst]) \n"
//                              :
//                              : [ src ] "r"(src8), [ dst ] "r"(dst8)
//                              : "f0");
//         src8 += 32;
//         dst8 += 32;
//         coord[lastDim]++; // += registerElems
//       } else {
//         __asm__ __volatile__("maskand m0, m0, m2 \n"
//                              "flw.ps f0, 0x0(%[src]) \n"
//                              "fsw.ps f0, 0x0(%[dst]) \n"
//                              "maskor m0, m0, m1 \n"
//                              :
//                              : [ src ] "r"(src8), [ dst ] "r"(dst8)
//                              : "f0");
//         unsigned int i = srcDimNum - 2;
//         src8 -= coord[lastDim] * typeSize;
//         dst8 -= coord[lastDim] * typeSize;
//         coord[lastDim] = 0;
//         if ((coord[i] + 1) != dstIndex[i]) {
//           coord[i]++;
//           offsetIn += actPitch[i];
//           offsetOut += dstPitch[i];
//           src8 += actPitch[i] * typeSize;
//           dst8 += dstPitch[i] * typeSize;
//         } else {
//           while ((coord[i] + 1) == dstIndex[i]) {
//             if (i != 0) {
//               coord[i] = 0;
//               offsetOut -= (dstIndex[i] - 1) * dstPitch[i];
//               offsetIn -= (actIndex[i] - 1) * actPitch[i];
//               src8 -= (actIndex[i] - 1) * typeSize * actPitch[i];
//               dst8 -= (dstIndex[i] - 1) * typeSize * dstPitch[i];
//               i--;
//               coord[i]++;
//               offsetOut += dstPitch[i];
//               offsetIn += actPitch[i];
//               src8 += actPitch[i] * typeSize;
//               dst8 += dstPitch[i] * typeSize;
//             } else {
//               done = true;
//               break;
//             }
//           }
//         }
//       }
//     }
//     if (! done) {
//       offsetOut += coord[lastDim]; // Due to addresser, last pitch = 1
//       while (not done && offsetOut < posMax) {
//         tOutput[offsetOut] = tAInput[offsetIn];
//         done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
//                           actPitch, dstPitch);
//       }
//     }
//   }
// }

GEN_INSTANCES_OP(template, fwdLibCopyInst, void *dst, void *dstDims, void *dstPitches,
                      void *src, void *srcDims, void *srcPitches, unsigned int srcDimNum,
                       float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibCopyInstThreaded, void *dst, void *dstDims, void *dstPitches,
                                  void *src, void *srcDims, void *srcPitches, unsigned int srcDimNum,
                                  float *scale, int32_t *offset,
                                  uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibCopyInstVectorized, void *dst, void *dstDims, void *dstPitches,
                                  void *src, void *srcDims, void *srcPitches,
                                  unsigned int srcDimNum,
                                  float *scale, int32_t *offset, uint64_t flags);
