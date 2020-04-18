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

#ifndef _ELEMENT_BOOL_INST_H_
#define _ELEMENT_BOOL_INST_H_

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

/**
 * @brief Given to tensors, it gives the result of the opType applied elementwise.
 *
 * Given an operator opType and two input tensors A, B, it generates the bool
 * tensor C in the following way @f$ C_{i,j} = opType(A_{i,j}, B_{i,j}) @f$. 
 * In this version all the work is done by the first minion.
 * 
 * @warning It comes without doubt that A and B must have the same dimensions.
 * 
 * @tparam srcType The type of the elements in the input tensors.
 * @tparam opType An operator that takes two srcType elements and returns a 
    bool (>, \geq, =, etc).
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
inline void fwdLibElementBoolInst(LibTensor* outT, LibTensor* in1T,
                                  LibTensor* in2T) {

  auto dstH = outT->getHandle<srcType>();
  auto srcH1 = in1T->getHandle<srcType>();
  auto srcH2 = in2T->getHandle<srcType>();

  bool* aDstT = reinterpret_cast<bool*>(dstH.getUnsafePtrdbg());
  srcType* srcT1 = reinterpret_cast<srcType*>(srcH1.getUnsafePtrdbg());
  srcType* srcT2 = reinterpret_cast<srcType*>(srcH2.getUnsafePtrdbg());
  
  // const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> aSrcT1(srcT1, srcH1.getScaledbg(), srcH1.getOffsetdbg());
  // const Addresser<srcType> aSrcT2(srcT2, scale[1], offset[1]);
  const Addresser<srcType> aSrcT2(srcT2, srcH2.getScaledbg(), srcH2.getOffsetdbg());
  // bool *aDstT = (bool *)dstT;

  // unsigned int *srcIndex = (unsigned int *)srcDims;
  dim_t srcIndex[max_tensor_dimensions] = {0,};
  srcH1.cpydims(srcIndex);

  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  dim_t dstPitch[max_tensor_dimensions] = {0,};
  dstH.cpypitchesdbg(dstPitch); 
  // unsigned int *src1Pitch = (unsigned int *)src1Pitches;
  dim_t src1Pitch[max_tensor_dimensions] = {0,};
  srcH1.cpypitchesdbg(src1Pitch); 
  // unsigned int *src2Pitch = (unsigned int *)src2Pitches;
  dim_t src2Pitch[max_tensor_dimensions] = {0,};
  srcH1.cpypitchesdbg(src2Pitch);
  unsigned int srcDimNum = static_cast<unsigned int>(srcH1.getNumDimsdbg());
  
  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc1Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc2Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrc1Pitch[i] = src1Pitch[i];
    eSrc2Pitch[i] = src2Pitch[i];
  }

  uint64_t addrSrc1, addrSrc2, addrDst;

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, opType> op;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc1 = x * eSrc1Pitch[0] + y * eSrc1Pitch[1] +
                         z * eSrc1Pitch[2] + w * eSrc1Pitch[3] +
                         q * eSrc1Pitch[4] + r * eSrc1Pitch[5];
              addrSrc2 = x * eSrc2Pitch[0] + y * eSrc2Pitch[1] +
                         z * eSrc2Pitch[2] + w * eSrc2Pitch[3] +
                         q * eSrc2Pitch[4] + r * eSrc2Pitch[5];
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
 * Given an operator opType and two input tensors A, B, it generates the bool
 * tensor C in the following way @f$ C_{i,j} = opType(A_{i,j}, B_{i,j}) @f$. 
 * This is the threaded version. 
 *
 * @note This implementation is similar to the CopyInstThreaded, where the
 *  code is more explained.
 * 
 * @warning It comes without doubt that A and B must have the same dimensions.
 * 
 * @tparam srcType The type of the elements in the input tensors.
 * @tparam opType An operator that takes two srcType elements and returns a 
    bool.
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
template <typename srcType, typename opType>
inline void fwdLibElementBoolInstThreaded(LibTensor* outT, LibTensor* in1T,
                                          LibTensor* in2T, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  auto dstH = outT->getHandle<srcType>();
  auto srcH1 = in1T->getHandle<srcType>();
  auto srcH2 = in2T->getHandle<srcType>();

  void* dstT = reinterpret_cast<void*>(dstH.getUnsafePtrdbg());
  srcType* srcT1 = reinterpret_cast<srcType*>(srcH1.getUnsafePtrdbg());
  srcType* srcT2 = reinterpret_cast<srcType*>(srcH2.getUnsafePtrdbg());
 
  // const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> aSrcT1(srcT1, srcH1.getScaledbg(), srcH1.getOffsetdbg());
  // const Addresser<srcType> aSrcT2(srcT2, scale[1], offset[1]);
  const Addresser<srcType> aSrcT2(srcT2, srcH2.getScaledbg(), srcH2.getOffsetdbg());
  bool *aDstT = reinterpret_cast<bool*>(dstT);
  
  // unsigned int *actIndex = (unsigned int *)srcDims;
  dim_t actIndex[max_tensor_dimensions] = {0,};
  srcH1.cpydims(actIndex);

  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  dim_t dstPitch[max_tensor_dimensions] = {0,};
  dstH.cpypitchesdbg(dstPitch);
  // unsigned int *act1Pitch = (unsigned int *)src1Pitches;
  dim_t act1Pitch[max_tensor_dimensions] = {0,};
  srcH1.cpypitchesdbg(act1Pitch); 
  // unsigned int *act2Pitch = (unsigned int *)src2Pitches;
  dim_t act2Pitch[max_tensor_dimensions] = {0,};
  srcH2.cpypitchesdbg(act2Pitch);
  unsigned int srcDimNum = static_cast<unsigned int>(srcH1.getNumDimsdbg());
  
  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
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
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

/**
 * @brief Given two tensors, it gives the result of the opType applied elementwise.
 *
 * Given an operator opType and two input tensors A, B, it generates a bool
 * tensor C in the following way @f$ C_{i,j} = opType(A_{i,j}, B_{i,j}) @f$. 
 * This is the threaded and and vectorized version of the operator.
 *
 * @note This implementation is similar to the CopyInstVectorized, where the
 *  code is more explained.
 * 
 * @warning It comes without doubt that A and B must have the same dimensions.
 * 
 * @tparam srcType The type of the elements in the input tensors.
 * @tparam opType An operator that takes two srcType elements and returns a 
    bool.
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
template <typename src1Type, typename src2Type, typename opType>
inline void fwdLibElementBoolInstVectorized(LibTensor* outT, LibTensor* in1T,
                                            LibTensor* in2T, const float* scale,
                                            const int32_t* offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  auto srcH1 = in1T->getHandle<src1Type>();
  auto srcH2 = in2T->getHandle<src2Type>();

  void* dstT = reinterpret_cast<void*>(outT->getUnsafePtr());
  src1Type* srcT1 = reinterpret_cast<src1Type*>(srcH1.getUnsafePtrdbg());
  src2Type* srcT2 = reinterpret_cast<src2Type*>(srcH2.getUnsafePtrdbg());

  // unsigned int *actIndex = (unsigned int *)srcDims;
  dim_t actIndex[max_tensor_dimensions] = {0,};
  srcH1.cpydims(actIndex);

  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  dim_t dstPitch[max_tensor_dimensions] = {0,};
  outT->dbgcpypitches(dstPitch); 
  // unsigned int *actPitch = (unsigned int *)src1Pitches;
  dim_t actPitch[max_tensor_dimensions] = {0,};
  srcH1.cpypitchesdbg(actPitch); 
  // bool *dstAddr = (bool *)dstT;
  bool* dstAddr = reinterpret_cast<bool*>(dstT);
  
  uintptr_t srcAddr1 = (uintptr_t)srcT1;
  uintptr_t srcAddr2 = (uintptr_t)srcT2;

  unsigned int srcDimNum = static_cast<unsigned int>(srcH1.getNumDimsdbg());
  
  Operator<Addresser<src1Type>, Addresser<src2Type>, Addresser<src2Type>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<src1Type>();
  getCachelinePartition(1, numElemsDst, initialAddr, maxRead,
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

  int32_t gatherValues[8];
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
    dstAddr += offsetOut;

    unsigned int cnt = 0;
    while(cnt < registersInRow) {
      __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
      op.doOpVect(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset);
      cnt++;
      srcAddr1 += 8 * typeSize;
      srcAddr2 += 8 * typeSize;
      dstAddr += 8;
    }
    if (res > 0) {
      __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      op.doOpVect(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset);
    }
    if (lastRow)
      return;

    dstAddr = (bool *)dstT;
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
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ELEMENT_BOOL_INST_H_

