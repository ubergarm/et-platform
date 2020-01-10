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

/**
 * @brief Copies the src matrix to the dst matrix.
 *
 * It makes a copy of the tensor src into the dst tensor, which may not
 * have the same pitches or dimensions, so it allows a reshaping. In this
 * version all the work is done by the same minion.
 * 
 * @tparam srcType The type of the elements in the tensor.
 * @param[out] dst Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] src Pointer to the input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] srcPitches Vector of pitches of the input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] scale, offset Parameters for the quantization.
 */
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

/**
 * @brief Copies the src matrix to the dst matrix.
 *
 * It makes a copy of the tensor src into the dst tensor, which may not
 * have the same pitches or dimensions, so it allows a reshaping. This is
 * the threaded version for this operator, so several minions are used.
 * 
 * @tparam srcType The type of the elements in the tensor.
 * @param[out] dst Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] src Pointer to the input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] srcPitches Vector of pitches of the input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] scale, offset Parameters for the quantization.
 * @param[in] flags Gives the information of the Active Shires and the
 *  type of evict required.
 */
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

/**
 * @brief Copies the src matrix to the dst matrix.
 *
 * It makes a copy of the tensor src into the dst tensor, which may not
 * have the same pitches or dimensions, so it allows a reshaping. This is
 * the threaded and vectorized version for this operator.
 *
 * @Warning It is assumed that the destination tensor starts at the beginning
 *  of a cacheline.
 * 
 * @tparam srcType The type of the elements in the tensor.
 * @param[out] dst Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] src Pointer to the input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] srcPitches Vector of pitches of the input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] scale, offset Parameters for the quantization.
 * @param[in] flags Gives the information of the Active Shires and the
 *  type of evict required.
 * @param[in] minionOffset The first minion that is assigned to this node.
 * @param[in] assignedMinions Amount of minions avaliable.
 */
template <typename srcType>
void dnn_lib::fwdLibCopyInstVectorized(void *dst, void *dstDims,
                                       void *dstPitches, void *src,
                                       void *srcDims, void *srcPitches,
                                       unsigned int srcDimNum, float *scale,
                                       int32_t *offset, uint64_t flags,
                                       const uint32_t minionOffset,
                                       const uint32_t assignedMinions) {

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (32 * ACTIVE_SHIRES) : assignedMinions;
  if ((minionId >= activeMinions) || (minionId >= activeMinions))
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
  unsigned int k = 0;                  // Amount of non-zero coordinates
  unsigned int coord[srcDimNum]= {0}; // Vector of coordinates
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
  unsigned int registerElems = 0;
  if (laneElems != 0) {
    registerElems = 8 * laneElems;
  } else {
    registerElems = 4;
  }
  unsigned int maxRow = (srcDimNum > 1) ? posMax / dstPitch[lastDim - 1] : 0;
  unsigned int elementsInRow = 0, registersInRow = 0, res = 0, spareElems = 0, fullLanes = 0;
  uint8_t mask = 0;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;

  if (srcDimNum == 1) {
    lastDim++;
    coord[0] = 0;
  }

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
      break;

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
                                  float *scale, int32_t *offset, uint64_t flags,
                                  const uint32_t minionOffset, const uint32_t assignedMinions);
