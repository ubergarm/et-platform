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
void dnn_lib::fwdLibInsertTensorInst(void *dst, void *dstDims, void *dstPitches,
                                     unsigned int dstDimNum, void *src2,
                                     void *src2Dims, void *src2Pitches,
                                     void *pcoord, unsigned int count,
                                     unsigned int axis, float *scale,
                                     int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tSmallInput(src2, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *smallIndex = (unsigned int *)src2Dims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *smallPitch = (unsigned int *)src2Pitches;

  unsigned int *coord = (unsigned int *)pcoord;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eOffsets[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < dstDimNum; i++) {
    eDims[i] = smallIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = smallPitch[i];
    eOffsets[i] = coord[i];
  }

  size_t advanceOnAxis = 0;

  for (size_t cnt = 0; cnt < count; cnt++) {
    // We can use this loop for all shapes.
    for (size_t x = 0; x < eDims[0]; x++) {
      for (size_t y = 0; y < eDims[1]; y++) {
        for (size_t z = 0; z < eDims[2]; z++) {
          for (size_t w = 0; w < eDims[3]; w++) {
            for (size_t q = 0; q < eDims[4]; q++) {
              for (size_t r = 0; r < eDims[5]; r++) {
                tOutput[(eOffsets[0] + x) * eDstPitch[0] +
                        (eOffsets[1] + y) * eDstPitch[1] +
                        (eOffsets[2] + z) * eDstPitch[2] +
                        (eOffsets[3] + w) * eDstPitch[3] +
                        (eOffsets[4] + q) * eDstPitch[4] +
                        (eOffsets[5] + r) * eDstPitch[5] + advanceOnAxis] =
                    tSmallInput[x * eSrcPitch[0] + y * eSrcPitch[1] +
                                z * eSrcPitch[2] + w * eSrcPitch[3] +
                                q * eSrcPitch[4] + r * eSrcPitch[5]];
              }
            }
          }
        }
      }
    }
    advanceOnAxis += eDstPitch[axis] * eDims[axis];
  }
}

//FIXME This version fits the small cases that currently are not vectorized,
//but it still fails some tests.
//template <typename srcType>
//void dnn_lib::fwdLibInsertTensorInstThreaded(void *dst, void *dstDims,
//                                             void *dstPitches,
//                                             unsigned int dstDimNum, void *src2,
//                                             void *src2Dims, void *src2Pitches,
//                                             void *poffsets, unsigned int count,
//                                             unsigned int axis, float *scale,
//                                             int32_t *offset, uint64_t flags) {
//  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
//  const Addresser<srcType> tAInput(src2, scale[0], offset[0]);
//
//  unsigned int *dstIndex = (unsigned int *)dstDims;
//  unsigned int *actIndex = (unsigned int *)src2Dims;
//
//  unsigned int *actPitch = (unsigned int *)src2Pitches;
//  unsigned int *dstPitch = (unsigned int *)dstPitches;
//  unsigned int *coord = (unsigned int *)poffsets;
//
//  unsigned int minionId = get_minion_id();
//  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
//  if (minionId >= activeMinions)
//    return;
//  size_t typeSize = getsize<srcType>();
//  unsigned int cll = 64/typeSize;
//
//  //Computing initial and last Address for each minion
//  unsigned int helper = actIndex[axis];
//  actIndex[axis] *= count;
//  unsigned int lastPos, addrOffset;
//  if(axis == dstDimNum - 1) {
//    //TODO these two for should be merged in one.
//    //FIXME It should depend on axis == n-1, where the lastPos should be in the
//    //next last element and in the axis != n-1, where the last element is as
//    //here
//    addrOffset = lastPos =  0;
//    for (unsigned int i = 0; i < dstDimNum; i++) {
//      addrOffset += coord[i] * dstPitch[i];
//    }
//    // Is it really necessary the last dimension term?
//    for (unsigned int i = dstDimNum - 2; i < dstDimNum - 1; i++) {
//      lastPos += (coord[i] + actIndex[i]) *
//               dstPitch[i];
//    }
//    for (unsigned int i = 0; i < dstDimNum - 1; i++) {
//      lastPos += (coord[i] + actIndex[i] - 1) * dstPitch[i];
//    }
//  }
//  else {
//    lastPos = (coord[axis] + actIndex[axis]) * dstPitch[axis];
//    addrOffset = coord[axis] * dstPitch[axis];
//    for (unsigned int i = 0; i < axis; i++) {
//      lastPos += (coord[i] + actIndex[i] - 1) * dstPitch[i];
//      addrOffset += coord[i] * dstPitch[i];
//    }
//  }
//
//  unsigned int moved = addrOffset % cll;
//  unsigned int ncl = (moved + lastPos - addrOffset - 1) / cll + 1;
//  unsigned int mcl = (ncl - 1) / activeMinions + 1;
//  unsigned int div = ncl / mcl;
//  unsigned int maxRead;
//  if (minionId < div)
//    maxRead = mcl * cll;
//  else if (minionId == div)
//    maxRead = (ncl - div * mcl) * cll;
//  else
//    return;
//  unsigned int addrOut = addrOffset + maxRead * minionId;
//  unsigned int posMax = std::min(addrOut + maxRead - moved, lastPos);
//  if (minionId != 0){
//    addrOut -= moved;
//  }
//
//  //Jumping to the next useful position
//  unsigned int coordIn[dstDimNum], k, addrIn;
//  getNonPaddingCoordinates(coordIn, addrOut - addrOffset, dstDimNum, dstPitch, actIndex, k);
//  addrIn = addrOut = 0;
//  for (unsigned int i = 0; i < axis; i++) {
//    addrOut += (coord[i] + coordIn[i]) * dstPitch[i];
//    addrIn += coordIn[i] * actPitch[i];
//  }
//  addrOut += (coord[axis] + coordIn[axis]) * dstPitch[axis];
//  addrIn += (coordIn[axis] % helper) * actPitch[axis];
//  for (unsigned int i = axis + 1; i < dstDimNum; i++) {
//    addrIn += coordIn[i] * actPitch[i];
//    addrOut += (coord[i] + coordIn[i]) * dstPitch[i];
//  }
//
//  bool done = false;
//  while ((addrOut < posMax) && !done) {
//    tOutput[addrOut] = tAInput[addrIn];
//  // TODO try using two getoffsets functions in order to verify this is correct
//    for (int j = dstDimNum - 1; j >= 0; j--) {
//      if (coordIn[j] != (actIndex[j] - 1)) {
//        addrOut += dstPitch[j];
//        coordIn[j]++;
//        //if ((j != axis) || (coordIn[axis] % helper != 0))
//        addrIn += actPitch[j];
//        //TODO avoid this if and module every iteration with a counter
//        if ((j == axis) && (coordIn[axis] % helper == 0))
//          addrIn -= helper * actPitch[axis];
//        break;
//      } else if (j != 0) {
//        if (j != axis)
//          addrIn -= (actIndex[j] - 1) * actPitch[j];
//        else
//          addrIn -= (helper - 1) * actPitch[axis];
//        addrOut -= (actIndex[j] - 1) * dstPitch[j];
//        coordIn[j] = 0;
//      } else
//        done = true;
//    }
//  }
//}

template <typename srcType>
inline void insertRow(uint8_t *dst, uint8_t *src, const unsigned int& addrOut,
                      const unsigned int& addrIn, const int32_t& typeSize,
                      int lanes, int res, int32_t *gatherValues) {
  uint8_t *dst8 = (uint8_t *) dst + addrOut * typeSize;
  uint8_t *src8 = (uint8_t *) src + addrIn * typeSize;
  __asm__ __volatile__("mov.m.x m0, zero, 0xff");
  while (lanes > 8) {
    __asm__ __volatile__("flw.ps f0, 0x0(%[src])\n"
                         "fsw.ps f0, 0x0(%[dst])\n"
                         :
                         : [ src ] "r"(src8), [ dst ] "r"(dst8)
                         : "f0", "memory");
    lanes -= 8;
    src8 += 32;
    dst8 += 32;
  }
  __asm__ __volatile__(
  "maskand m0, m1, m1\n"
  "flw.ps f0, 0x0(%[src])\n"
  "fsw.ps f0, 0x0(%[dst])\n"
  :
  : [ src ] "r"(src8), [ dst ] "r"(dst8)
  : "f0", "memory");
  src8 += 4*lanes;
  dst8 += 4*lanes;
  if (res != 0) {
    if (getsize<srcType>() == 2) {
      __asm__ __volatile__(
          "maskand m0, m2, m2\n"
          "flw.ps f1, 0x0(%[gatherValues]) \n"
          "fgh.ps f0, f1(%[src]) \n"
          "fsch.ps f0, f1(%[dst]) \n"
          :
          : [ src ] "r"(src8), [ dst ] "r"(dst8),
            [ gatherValues ] "r"(gatherValues)
          : "f0", "f1", "memory");
    } else if (getsize<srcType>() == 1) {
      __asm__ __volatile__(
          "maskand m0, m2, m2\n"
          "flw.ps f1, 0x0(%[gatherValues]) \n"
          "fgb.ps f0, f1(%[src]) \n"
          "fscb.ps f0, f1(%[dst]) \n"
          :
          : [ src ] "r"(src8), [ dst ] "r"(dst8),
            [ gatherValues ] "r"(gatherValues)
          : "f0", "f1", "memory");
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibInsertTensorInstThreaded(void *dst, void *dstDims,
                                             void *dstPitches,
                                             unsigned int dstDimNum, void *src2,
                                             void *src2Dims, void *src2Pitches,
                                             void *poffsets, unsigned int count,
                                             unsigned int axis, float *scale,
                                             int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  int32_t typeSize = (int32_t) getsize<srcType>();
  unsigned int cll = 64/typeSize;

  if ((dstDimNum >= 2) && (dstPitch[dstDimNum - 2]%cll != 0)) {
    fwdLibInsertTensorInst<srcType>(dst, dstDims, dstPitches,
                                     dstDimNum, src2,
                                     src2Dims, src2Pitches,
                                     poffsets, count,
                                     axis, scale,
                                     offset);
    return;
  }

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src2, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)src2Dims;

  unsigned int *actPitch = (unsigned int *)src2Pitches;
  unsigned int *coord = (unsigned int *)poffsets;

  // We compute the offset address
  unsigned int offsetNum = coord[0] * dstPitch[0];
  for (unsigned int i = 1; i < dstDimNum; i++)
    offsetNum += coord[i] * dstPitch[i]; // Offset Address
  unsigned int jump = dstPitch[axis] * actIndex[axis];

  unsigned int dimRow = 0;
  if (dstDimNum > 1)
    dimRow = dstDimNum - 2;
  unsigned int lastDim = dstDimNum - 1;

  int32_t gatherValues[8] = { 0, typeSize, 2 * typeSize, 3 * typeSize,
                              4 * typeSize, 5 * typeSize, 6 * typeSize,
                              7 * typeSize};

  int lanes, res;
  getLanesResTView<srcType>(lanes, res, actIndex[lastDim]);

  uint32_t mask = (1 << (((lanes - 1) % 8) + 1)) - 1;
  __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                        :
                        : [ mask ] "r" (mask));
  mask = (1 << res) - 1;
  __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                        :
                        : [ mask ] "r" (mask));

  if (axis != lastDim) {
    unsigned int auxNRows = count * actIndex[0];
    for (int i = 1; i < lastDim; i++)
      auxNRows *= actIndex[i];
    unsigned int mRows = auxNRows / activeMinions;
    unsigned int mod = auxNRows - activeMinions * mRows;
    if (minionId < mod) {
      ++mRows;
      mod = 0;
    }
    if (unlikely(mRows == 0))
      return; // No work to do

    auxNRows /= count;
    unsigned int aux = (mod + mRows * minionId) / auxNRows;
    offsetNum += jump * aux;
    unsigned int initialAddrIn = ((mod + mRows * minionId) - aux * auxNRows) * actPitch[dimRow];

    unsigned int offsetIn[dstDimNum], offsetOut[dstDimNum];
    unsigned int initialAddr = offsetNum;
    getCoordinates(offsetIn, initialAddrIn, dstDimNum, actPitch);
    getCoordinates(offsetOut, initialAddr, dstDimNum, dstPitch);

    unsigned int addrOut = 0;
    for (int i = lastDim; i >= 0; i--) {
      offsetOut[i] += offsetIn[i];
      addrOut += dstPitch[i] * offsetOut[i];
    }
    bool done = false;
    while (mRows > 0) {
      insertRow<srcType>((uint8_t *) dst, (uint8_t *) src2, addrOut,
                initialAddrIn, typeSize, lanes, res, gatherValues);
      for (int j = dimRow; j >= 0; j--) {
        if (likely(offsetIn[j] != (actIndex[j] - 1))) {
          initialAddrIn += actPitch[j];
          addrOut += dstPitch[j];
          offsetIn[j]++;
          break;
        } else if (likely(j != 0)){
          initialAddrIn -= (actIndex[j] - 1) * actPitch[j];
          addrOut -= (actIndex[j] - 1) * dstPitch[j];
          offsetIn[j] = 0;
        } else {
          initialAddrIn = offsetIn[j]  = 0;
          offsetNum += jump;
          addrOut = offsetNum;
        }
      }
      mRows--;
    }
  } else {
    unsigned int auxNRows = actIndex[0];
    for (int i = 1; i < dstDimNum - 1; i++)
      auxNRows *= actIndex[i];

    if (auxNRows > activeMinions) {
      unsigned int mRows = auxNRows / activeMinions;
      unsigned int mod = auxNRows - activeMinions * mRows;
      unsigned int initialAddrIn;
      // We add to the initial address the new address in the tensor
      if (minionId < mod) {
        ++mRows;
        initialAddrIn = mRows * actPitch[dimRow] * minionId;
      } else
        initialAddrIn = (mod + minionId * mRows) * actPitch[dimRow];
      unsigned int k, offsetIn[dstDimNum], offsetOut[dstDimNum];
      getNonPaddingCoordinates(offsetIn, initialAddrIn, dstDimNum, actPitch,
                               actIndex, k);
      getNonPaddingCoordinates(offsetOut, offsetNum, dstDimNum, dstPitch,
                               dstIndex, k);
      unsigned int addrOut = 0;
      for (int i = dstDimNum - 1; i >= 0; i--) {
        offsetOut[i] += offsetIn[i];
        addrOut += dstPitch[i] * offsetOut[i];
      }
      for (int i = 0; i < mRows; i++) {
        for (int j = 0; j < count; j++) {
          insertRow<srcType>((uint8_t *) dst, (uint8_t *) src2, addrOut,
                    initialAddrIn, typeSize, lanes, res, gatherValues);
          addrOut += actIndex[axis] * dstPitch[axis];
        }
        addrOut -= count * actIndex[axis] * dstPitch[axis];
        for (int j = dimRow; j >= 0; j--) {
          if (offsetIn[j] != (actIndex[j] - 1)) {
            addrOut += dstPitch[j];
            initialAddrIn += actPitch[j];
            offsetIn[j]++;
            break;
          } else {
            addrOut -= (actIndex[j] - 1) * dstPitch[j];
            initialAddrIn -= (actIndex[j] - 1) * actPitch[j];
            offsetIn[j] = 0;
          }
        }
      }
    } else {
      unsigned int mperRow = activeMinions / auxNRows;
      if (minionId >= mperRow * auxNRows)
        return;
      unsigned int rowtomin = minionId / mperRow;

      unsigned int offsetOut[dstDimNum];
      for (unsigned int i = 0; i < dstDimNum; i++) {
        offsetOut[i] = coord[i];
      }

      if(axis > 0) {
        unsigned int falsepitch[axis];
        falsepitch[dimRow] = 1;
        for (int i = dimRow; i > 0; i--)
          falsepitch[i - 1] = falsepitch[i] * actIndex[i];

        for (int i = 0; i < axis; i++) {
          unsigned int aux = rowtomin / falsepitch[i];
          offsetOut[i] += aux;
          rowtomin -= aux * falsepitch[i];
        }
      }
      unsigned int addrOut = 0;
      for (int i = axis; i >= 0; i--) {
        addrOut += dstPitch[i] * offsetOut[i];
      }
      unsigned int lastRowElem = addrOut + actIndex[axis] * dstPitch[axis] * count;
      unsigned int save = addrOut;
      unsigned int cll = 64 / getsize<srcType>();
      unsigned int modulo = addrOut % cll;
      //unsigned int maximalPos = jump * count;
      unsigned int clperRow = (modulo + (jump * count) - 1) / cll + 1;
      unsigned int mcl = clperRow / mperRow;
      unsigned int mod = clperRow - mperRow * mcl;
      unsigned int maxRead;
      unsigned int minmodule = minionId % mperRow;
      if (minmodule != 0) {
        addrOut -= modulo;
        if (minmodule < mod){
          ++mcl;
          addrOut += mcl * cll * minmodule;
        } else {
          addrOut += (mod + minmodule * mcl) * cll;
        }
        maxRead = mcl * cll;
      } else {
        if (mod != 0) {
          ++mcl;
        }
        maxRead = mcl * cll - modulo;
      }
      if (mcl == 0) {
        return;
      }
      //maximalPos += save - 1;
      unsigned int k;
      getNonPaddingCoordinates(offsetOut, addrOut, dstDimNum, dstPitch,
                               dstIndex, k);
      addrOut = 0;
      for (unsigned int i = 0; i < dstDimNum; i++) {
        addrOut += offsetOut[i] * dstPitch[i];
      }

      unsigned int offsetIn[dstDimNum];
      for (unsigned int i = 0; i < dstDimNum; i++) {
        offsetIn[i] = offsetOut[i] - coord[i];
      }
      offsetIn[axis] = offsetIn[axis] % actIndex[axis];
      unsigned int initialAddrIn = 0;
      for (unsigned int i = 0; i < dstDimNum; i++) {
        initialAddrIn += offsetIn[i] * actPitch[i];
      }
      maxRead = std::min(maxRead, lastRowElem - addrOut);
      unsigned int length = std::min(maxRead, actIndex[axis] - offsetIn[axis]);
      int auxlanes, auxres;
      getLanesResTView<srcType>(auxlanes, auxres, length);
      maxRead -= length;
      mask = (1 << (((auxlanes - 1) % 8) + 1)) - 1;
      __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      mask = (1 << auxres) - 1;
      __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));

      insertRow<srcType>((uint8_t *) dst, (uint8_t *) src2, addrOut,
                initialAddrIn, typeSize, auxlanes, auxres, gatherValues);
      addrOut += length * dstPitch[axis];
      initialAddrIn -= offsetIn[axis] * actPitch[axis];

      mask = (1 << (((lanes - 1) % 8) + 1)) - 1;
      __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      mask = (1 << res) - 1;
      __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      while (maxRead > actIndex[axis]) {
        insertRow<srcType>((uint8_t *) dst, (uint8_t *) src2, addrOut,
                  initialAddrIn, typeSize, lanes, res, gatherValues);
        maxRead -= actIndex[axis];
        addrOut += actIndex[axis] * dstPitch[axis];
      }
      getLanesResTView<srcType>(auxlanes, auxres, maxRead);
      mask = (1 << (((auxlanes - 1) % 8) + 1)) - 1;
      __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      mask = (1 << auxres) - 1;
      __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      insertRow<srcType>((uint8_t *) dst, (uint8_t *) src2, addrOut,
                initialAddrIn, typeSize, auxlanes, auxres, gatherValues);
    }
  }
}

GEN_INSTANCES_OP(template, fwdLibInsertTensorInst, void *dst, void *dstDims,
                               void *dstPitches, unsigned int dstDimNum,
                               void *src2, void *src2Dims, void *src2Pitches,
                               void * poffsets, unsigned int count,
                               unsigned int axis, float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibInsertTensorInstThreaded, void *dst, void *dstDims,
                                void *dstPitches, unsigned int dstDimNum,
                                void *src2, void *src2Dims, void *src2Pitches,
                                void * poffsets, unsigned int count,
                                unsigned int axis, float *scale, int32_t *offset, uint64_t flags);
