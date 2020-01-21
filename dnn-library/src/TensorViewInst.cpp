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
void dnn_lib::fwdLibTensorViewInst(void *dst, void *dstDims, void *dstPitches,
                                   unsigned int dstDimNum, void *src,
                                   void *srcDims, void *srcPitches,
                                   unsigned int srcDimNum, void *pcoord,
                                   float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  long unsigned int *coord = (long unsigned int *)pcoord;

  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);
  int offsetIn = 0;

  unsigned int eDstDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcCnt[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (int i = 0; i < MAX_TENSOR_DIMENSIONS; i++) {
    if (i < dstDimNum) {
      eDstDims[i] = dstIndex[i];
      eDstPitch[i] = dstPitch[i];
    }
    if (i < srcDimNum) {
      offsetIn += coord[i] * actPitch[i];
      eSrcCnt[i] = coord[i];
    }
  }
  bool done = false;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eDstDims[0]; x++) {
    for (size_t y = 0; y < eDstDims[1]; y++) {
      for (size_t z = 0; z < eDstDims[2]; z++) {
        for (size_t w = 0; w < eDstDims[3]; w++) {
          for (size_t q = 0; q < eDstDims[4]; q++) {
            for (size_t r = 0; r < eDstDims[5]; r++) {
              if (!done) {
                uint64_t addr = x * eDstPitch[0] + y * eDstPitch[1] +
                                z * eDstPitch[2] + w * eDstPitch[3] +
                                q * eDstPitch[4] + r * eDstPitch[5];
                tOutput[addr] = tAInput[offsetIn];
                for (int j = srcDimNum - 1; j >= 0; j--) {
                  if (eSrcCnt[j] != (actIndex[j] - 1)) {
                    offsetIn += actPitch[j];
                    eSrcCnt[j]++;
                    break;
                  } else if (j == 0) {
                    done = true;
                    break;
                  } else {
                    offsetIn -= (actIndex[j] - 1) * actPitch[j];
                    eSrcCnt[j] = 0;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibTensorViewInstThreaded(
    void *dst, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src, void *srcDims, void *srcPitches, unsigned int srcDimNum,
    void *pcoord, float *scale, int32_t *offset, uint64_t flags) {

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

  long unsigned int *coord = (long unsigned int *)pcoord;
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddrOut, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddrOut, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coordIn[srcDimNum];
  unsigned int coordOut[dstDimNum];
  unsigned int kOut = 0;
  getNonPaddingCoordinates(coordOut, initialAddrOut, dstDimNum, dstPitch,
                           dstIndex, kOut);

  unsigned int auxActPitch[srcDimNum];           //Pitch not accounting padding
  auxActPitch[srcDimNum - 1] = 1;
  for (int i = srcDimNum - 2; i >= 0; i--)
    auxActPitch[i] = auxActPitch[i + 1] * actIndex[i + 1];

  unsigned int auxdstPitch[dstDimNum];           //Pitch not accounting padding
  auxdstPitch[dstDimNum - 1] = 1;
  for (int i = dstDimNum - 2; i >= 0; i--)
    auxdstPitch[i] = auxdstPitch[i + 1] * dstIndex[i + 1];

  unsigned int addrIn = 0;
  unsigned int addrOut = 0;
  unsigned int elements_moved = 0;
  for (unsigned int j = 0; j < kOut; j++) {
    addrOut += dstPitch[j] * coordOut[j];
    elements_moved += auxdstPitch[j] * coordOut[j];
  }

  for (unsigned int i = 0; i < srcDimNum; i++) {
    coordIn[i] = elements_moved / auxActPitch[i];
    elements_moved = elements_moved - coordIn[i] * auxActPitch[i];
  }
  for (int i = srcDimNum - 1; i >= 0; i--) {
    coordIn[i] += (int)coord[i];
    if (coordIn[i] >= actIndex[i]) {
      coordIn[i] = coordIn[i] % actIndex[i];
      coordIn[i - 1] += 1;
    }
    addrIn += coordIn[i] * actPitch[i];
  }
  unsigned int posMax = maxRead + initialAddrOut;

  bool done = false;
  bool donein = false;
  while (!done && (addrOut < posMax)) {
    tOutput[addrOut] = tAInput[addrIn];
    donein = getOffsets(srcDimNum, coordIn, addrIn, actIndex, actPitch);
    done = getOffsets(dstDimNum, coordOut, addrOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddrOut, clperminion);
}

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
  if (getsize<srcType>() == 2) {
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

  } else if (getsize<srcType>() == 1) {
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

template <typename srcType>
void dnn_lib::fwdLibTensorViewInstVectorized(
    void *dst, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src, void *srcDims, void *srcPitches, unsigned int srcDimNum,
    void *pcoord, float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return; // Minion not working

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  long unsigned int *coord = (long unsigned int *)pcoord;
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddrOut, maxRead;
  int32_t typeSize = (int32_t) getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddrOut, maxRead,
                        minionId, activeMinions); // Obtain initial addr 4 minions
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
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddrOut, clperminion);
}

GEN_INSTANCES_OP(template, fwdLibTensorViewInst, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, void *src, void *srcDims,
                             void *srcPitches, unsigned int srcDimNum, void *poffsets,
                             float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibTensorViewInstThreaded, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, void *src, void *srcDims,
                             void *srcPitches, unsigned int srcDimNum, void *poffsets,
                             float *scale, int32_t *offset, uint64_t flags );
GEN_INSTANCES_OP(template, fwdLibTensorViewInstVectorized, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, void *src, void *srcDims,
                             void *srcPitches, unsigned int srcDimNum, void *poffsets,
                             float *scale, int32_t *offset, uint64_t flags );
