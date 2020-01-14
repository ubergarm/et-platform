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

inline __attribute__((always_inline)) void
getCoordinates(unsigned int iDx, unsigned int srcDimNum, unsigned int *sizes,
               unsigned int *coordinates) {

  unsigned int acum = sizes[srcDimNum - 1];
  coordinates[srcDimNum - 1] = iDx % acum;
  // it assumes sizes is never 0 fot the srcDimNum dimensions
  for (int dimI = srcDimNum - 2; dimI >= 0; dimI--) {
    if (iDx >= acum) {
      coordinates[dimI] = (iDx / acum) % (sizes[dimI]);
      acum *= sizes[dimI];
    } else {
      break;
    }
  }
}

inline __attribute__((always_inline)) uint64_t
calcAddrOffset(unsigned int *pitches, unsigned int *coordinates) {
  return (pitches[0] * coordinates[0]) + (pitches[1] * coordinates[1]) +
         (pitches[2] * coordinates[2]) + (pitches[3] * coordinates[3]) +
         (pitches[4] * coordinates[4]) + (pitches[5] * coordinates[5]);
}

/**
 * @brief Given to tensors, it gives the result of the opType applied elementwise.
 *
 * Given an operator opType and two input tensors A, B, it generates the srcType
 * tensor C in the following way @f$ C_{i,j} = opType(A_{i,j}, B_{i,j}) @f$. 
 * In this version all the work is done by the first minion.
 * 
 * @warning It comes without doubt that A and B must have the same dimensions.
 * 
 * @note TThe next version is a generalization of this function as it allows the 
 *  types of the three tensors not being the same.
 *
 * @tparam srcType The type of the elements in the input and output tensors.
 * @tparam opType An operator that takes two srcType elements and returns a 
    srcType (+, ·, etc).
 * @param[out] dstT Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] srcT1 Pointer to the first input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] src1Pitches Vector of pitches of the first input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] srcT2 Pointer to the second input matrix.
 * @param[in] src2Pitches Vector of pitches of the second input tensor.
 * @param[in] scale, offset Parameters for the quantization.
 */
template <typename srcType, typename opType>
void dnn_lib::fwdLibElementInst(void *dstT, void *dstDims, void *dstPitches,
                                void *srcT1, void *srcDims, void *src1Pitches,
                                unsigned int srcDimNum, void *srcT2,
                                void *src2Pitches, float *scale,
                                int32_t *offset) {

 unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> aSrcT2(srcT2, scale[1], offset[1]);
  Addresser<srcType> aDstT(dstT, scale[2], offset[2]);

  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *act1Pitch = (unsigned int *)src1Pitches;
  unsigned int *act2Pitch = (unsigned int *)src2Pitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc1Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc2Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrc1Pitch[i] = act1Pitch[i];
    eSrc2Pitch[i] = act2Pitch[i];
  }

  uint64_t addrSrc1, addrSrc2, addrDst;

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, opType> op;

  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc1 = x * eSrc1Pitch[0] + y * eSrc1Pitch[1] + z * eSrc1Pitch[2] +
                        w * eSrc1Pitch[3] + q * eSrc1Pitch[4] + r * eSrc1Pitch[5];
              addrSrc2 = x * eSrc2Pitch[0] + y * eSrc2Pitch[1] + z * eSrc2Pitch[2] +
                        w * eSrc2Pitch[3] + q * eSrc2Pitch[4] + r * eSrc2Pitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              op.doOp(aDstT, aSrcT1, aSrcT2, addrDst, addrSrc1, addrSrc2);
            }
          }
        }
      }
    }
  }
}


/**
 * @brief Given to tensors, it gives the result of the opType applied elementwise.
 *
 * Given an operator opType and two input tensors A, B, it generates the
 * tensor C in the following way @f$ C_{i,j} = opType(A_{i,j}, B_{i,j}) @f$. 
 * This is the threaded version.
 * 
 * @warning It comes without doubt that A and B must have the same dimensions.
 * 
 * @note This implementation is similar to the CopyInstThreaded, where the
 *  code is more explained.
 *
 * @note This is a generalization of the previous version as it allows the 
 *  types of the three tensors not being the same.
 *
 * @tparam src1Type The type of the elements in the first input tensors.
 * @tparam src2Type The type of the elements in the second input tensors.
 * @tparam dstType The type of the elements in the output tensor.
 * @tparam opType An operator that takes two srcType elements and returns a 
    srcType (+, ·, etc).
 * @param[out] dstT Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] srcT1 Pointer to the first input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] src1Pitches Vector of pitches of the first input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] srcT2 Pointer to the second input matrix.
 * @param[in] src2Pitches Vector of pitches of the second input tensor.
 * @param[in] scale, offset Parameters for the quantization.
 * @param[in] flags Controls the active shires and the type of evict that 
 *  should be done at the end of the function.
 */
template <typename src1Type, typename src2Type, typename dstType, typename opType>
void dnn_lib::fwdLibElementInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *src1Pitches, unsigned int srcDimNum, void *srcT2, void *src2Pitches, float *scale,
    int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<src1Type> aSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<src2Type> aSrcT2(srcT2, scale[1], offset[1]);
  Addresser<dstType> aDstT(dstT, scale[2], offset[2]);

  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *act1Pitch = (unsigned int *)src1Pitches;
  unsigned int *act2Pitch = (unsigned int *)src2Pitches;

  Operator<Addresser<src1Type>, Addresser<src2Type>, Addresser<dstType>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<src2Type>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn1 = 0;
  uint64_t offsetIn2 = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn1 += act1Pitch[j] * coord[j];
    offsetIn2 += act2Pitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    op.doOp(aDstT, aSrcT1, aSrcT2, offsetOut, offsetIn1, offsetIn2);
    done = getOffsets(srcDimNum, coord, offsetIn1, offsetIn2, offsetOut, actIndex,
                      act1Pitch, act2Pitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}



/**
 * @brief Given two tensors, it gives the result of the opType applied elementwise.
 *
 * Given an operator opType and two input tensors A, B, it generates a
 * tensor C in the following way @f$ C_{i,j} = opType(A_{i,j}, B_{i,j}) @f$. 
 * This is the threaded and and vectorized version of the operator.
 * 
 * @note This implementation is similar to the CopyInstVectorized, where the
 *  code is more explained.
 * 
 * @warning It comes without doubt that A and B must have the same dimensions.
 * 
* @note This is a generalization of the previous version as it allows the 
 *  types of the three tensors not being the same.
 *
 * @tparam src1Type The type of the elements in the first input tensors.
 * @tparam src2Type The type of the elements in the second input tensors.
 * @tparam dstType The type of the elements in the output tensor.
 * @tparam opType An operator that takes two srcType elements and returns a 
    srcType (+, ·, etc).
 * @param[out] dstT Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] srcT1 Pointer to the first input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] src1Pitches Vector of pitches of the first input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] srcT2 Pointer to the second input matrix.
 * @param[in] src2Pitches Vector of pitches of the second input tensor.
 * @param[in] scale, offset Parameters for the quantization.
 * @param[in] flags Controls the active shires and the type of evict that 
 *  should be done at the end of the function.
 */
template <typename src1Type, typename src2Type, typename dstType, typename opType>
void dnn_lib::fwdLibElementInstVectorized(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *src1Pitches, unsigned int srcDimNum, void *srcT2, void *src2Pitches,
    float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)src1Pitches;
  uintptr_t dstAddr = (uintptr_t)dstT;
  uintptr_t srcAddr1 = (uintptr_t)srcT1;
  uintptr_t srcAddr2 = (uintptr_t)srcT2;

  Operator<Addresser<src1Type>, Addresser<src2Type>, Addresser<dstType>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<src1Type>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;

  int32_t gatherValues[] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (unsigned int i = 0; i < 8; i++) {
      gatherValues[i] = i * typeSize;
  }

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
      elementsInRow = actIndex[lastDim] - coord[lastDim];
    } else if (coord[lastDim - 1] == maxRow) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
    } else {
      elementsInRow = actIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      res = elementsInRow - registersInRow * 8;
      mask = ((1 << res) - 1);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr1 += offsetIn * typeSize;
    srcAddr2 += offsetIn * typeSize;
    dstAddr += offsetOut * typeSize;
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

    unsigned int cnt = 0;

    while(cnt < registersInRow) {
      op.doOpVect(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset);
      cnt++;
      srcAddr1 += 8 * typeSize;
      srcAddr2 += 8 * typeSize;
      dstAddr += 8 * typeSize;
    }
    if (res > 0) {
      __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      op.doOpVect(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset);
    }

    if (lastRow)
      return;

    dstAddr = (uintptr_t)dstT;
    srcAddr1 = (uintptr_t)srcT1;
    srcAddr2 = (uintptr_t)srcT2;
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;
    done = getOffsets(lastDim , coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Add,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset);
GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Sub,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset);
GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Div,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset);
GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Mul,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset);
GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Max,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset);
GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Min,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset);
GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Pow,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Add,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches,float * scale,
                        int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Sub,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches,float * scale,
                        int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Div,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches,float * scale,
                        int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Mul,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches,float * scale,
                        int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Max,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches,float * scale,
                        int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Min,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches,float * scale,
                        int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Pow,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, float * scale ,
                                 int32_t  * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Add,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Sub,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Div,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Mul,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Max,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Min,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,float * scale,
                                 int32_t  * offset, uint64_t flags);
GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Pow,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, float * scale,
                                 int32_t  * offset, uint64_t flags);
