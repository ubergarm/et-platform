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
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

#include "Addresser.h"
#include "Float16.h"
#include "LibTensor.h"
#include "utils.h"

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
 * @tparam srcNElK The type of the elements in the nth input tensor
 * @tparam opType The operation to perform, returning a bool (>, \geq, =, etc).
 * @param[out] outT pointer to the output LibTensor.
 * @param[in] in1T pointer to first source LibTensor
 * @param[in] in2T pointer to second source LibTensor
 */
template <ElemKind src1ElK, ElemKind src2ElK, typename opType>
INLINE_ATTR void fwdLibElementBoolInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                       [[maybe_unused]] uint64_t flags, const uint32_t minionOffset = 0,
                                       [[maybe_unused]] const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* srcT1 = in1T->getRawDataPointer();
  void* srcT2 = in2T->getRawDataPointer();

  const Addresser<src1ElK> aSrcT1(srcT1, in1T->getScale(), in1T->getOffset());
  const Addresser<src2ElK> aSrcT2(srcT2, in2T->getScale(), in2T->getOffset());
  bool* aDstT = outT->getRawDataPointer<bool>();
  
  const dim_t *srcIndex = in1T->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *src1Pitch = in1T->strides().data();
  const dim_t *src2Pitch = in2T->strides().data();

  dim_t srcDimNum = in1T->ndims();

  size_t eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  size_t eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  size_t eSrc1Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  size_t eSrc2Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (dim_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrc1Pitch[i] = src1Pitch[i];
    eSrc2Pitch[i] = src2Pitch[i];
  }

  uint64_t addrSrc1, addrSrc2, addrDst;

  Operator<Addresser<src1ElK>, Addresser<src2ElK>, Addresser<BoolTy>, opType> op;
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
 * @tparam srcNElK The type of the elements in the nth input tensor
 * @tparam opType The operation to perform, returning a bool (>, \geq, =, etc).
 * @param[out] dstT Pointer to the output matrix.
 * @param[in] in1T pointer to first source LibTensor
 * @param[in] in2T pointer to second source LibTensor
 * @param[in] flags Controls the active shires and the type of evict that 
 *  should be done at the end of the function.
 */
template <ElemKind src1ElK, ElemKind src2ElK, typename opType>
INLINE_ATTR void fwdLibElementBoolInstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,
                                               const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  //  using src1Type = typename elemKind2elemTy<src1ElK>::type;
  //  using src2Type = typename elemKind2elemTy<src2ElK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* srcT1 = in1T->getRawDataPointer();
  void* srcT2 = in2T->getRawDataPointer();
  void* dstT = outT->getRawDataPointer();

  const Addresser<src1ElK> aSrcT1(srcT1, in1T->getScale(), in1T->getOffset());
  const Addresser<src2ElK> aSrcT2(srcT2, in2T->getScale(), in2T->getOffset());
  auto aDstT = outT->getRawDataPointer<bool>();

  const dim_t *actIndex = in1T->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *act1Pitch = in1T->strides().data();
  const dim_t *act2Pitch = in2T->strides().data();

  dim_t srcDimNum = in1T->ndims();

  Operator<Addresser<src1ElK>, Addresser<src2ElK>, Addresser<BoolTy>, opType> op;

  auto numElemsDst = dstPitch[0] * actIndex[0];

  size_t initialAddr, maxRead;
  size_t typeSize = getsize<bool>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  dim_array_t coord = {0};
  dim_t k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  size_t offsetIn1 = 0;
  size_t offsetIn2 = 0;
  size_t offsetOut = 0;
  for (size_t j = 0; j < k; j++) {
    offsetIn1 += act1Pitch[j] * coord[j];
    offsetIn2 += act2Pitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  auto posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    op.doOp(aDstT, aSrcT1, aSrcT2, offsetOut, offsetIn1, offsetIn2);
    done = getOffsets(srcDimNum, coord, offsetIn1, offsetIn2, offsetOut, actIndex,
                      act1Pitch, act2Pitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
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
 * @tparam srcNElK The type of the elements in the nth input tensor
 * @tparam opType The operation to perform, returning a bool (>, \geq, =, etc).
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
template <ElemKind src1ElK, ElemKind src2ElK, typename opType>
INLINE_ATTR void fwdLibElementBoolInstVectorized(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,
                                                 const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  const float scale[] = { in1T->getScale(), in2T->getScale(), outT->getScale()};
  const int32_t offset[] = {in1T->getOffset(), in2T->getOffset(), outT->getOffset()};
  
  using src1Type = typename elemKind2elemTy<src1ElK>::type;
  //  using src2Type = typename elemKind2elemTy<src2ElK>::type;
  

  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer();
  void* srcT1 = in1T->getRawDataPointer();
  void* srcT2 = in2T->getRawDataPointer();

  // unsigned int *actIndex = (unsigned int *)srcDims;
  const dim_t *actIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)src1Pitches;
  const dim_t *actPitch = in1T->strides().data();
  // bool *dstAddr = (bool *)dstT;
  bool* dstAddr = outT->getRawDataPointer<bool>();
  
  uintptr_t srcAddr1 = (uintptr_t)srcT1;
  uintptr_t srcAddr2 = (uintptr_t)srcT2;

  dim_t srcDimNum = in1T->ndims();

  Operator<Addresser<src1ElK>, Addresser<src2ElK>, Addresser<src2ElK>, opType> op;

  auto numElemsDst = dstPitch[0] * actIndex[0];

  size_t initialAddr, maxRead;
  size_t typeSize = getsize<src1Type>();
  getCachelinePartition(1 /* output size; data is boolean, uses int8 internally */, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  dim_array_t coord = {0};
  dim_t k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  size_t offsetIn = 0;
  size_t offsetOut = 0;
  for (size_t j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  auto posMax = maxRead + initialAddr;
  bool done = false;
  auto lastDim = srcDimNum - 1;

  int32_t gatherValues[8];
  for (int i = 0; i < 8; i++) {
    gatherValues[i] = static_cast<int32_t>(i * typeSize);
  }
  size_t maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  size_t elementsInRow, registersInRow, resVect, resWord;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  coord[0] *= (srcDimNum != 1);

  while (!done && (offsetOut < posMax)) {
    if (firstRow && (srcDimNum > 1) && coord[lastDim - 1] != maxRow) {
      elementsInRow = actIndex[lastDim] - coord[lastDim];
    } else if ((srcDimNum == 1) || (coord[lastDim - 1] == maxRow)) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
      if (elementsInRow > (actIndex[lastDim] - coord[lastDim])) {
        elementsInRow = actIndex[lastDim] - coord[lastDim];
      }
    } else {
      elementsInRow = actIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      resVect = elementsInRow - registersInRow * 8;
      resWord = resVect >= 4 ? resVect - 4 : resVect;
      mask = static_cast<uint8_t>((1UL << resWord) - 1);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr1 += offsetIn * typeSize;
    srcAddr2 += offsetIn * typeSize;
    dstAddr += offsetOut;

    unsigned int cnt = 0;
    while (cnt < registersInRow) {
      // TODO : when moving to clang, remove following line and pass 0xff as extra last parameter
      __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
      op.doOpVect(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset);
      cnt++;
      srcAddr1 += 8 * typeSize;
      srcAddr2 += 8 * typeSize;
      dstAddr += 8;
    }

    if (resVect >= 4) {
      // TODO : when moving to clang, remove following line and pass 0x0f as extra last parameter
      __asm__ __volatile__("mov.m.x m0, zero, 0x0f \n");
      op.doOpVect(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset);
      srcAddr1 += 4 * typeSize;
      srcAddr2 += 4 * typeSize;
      dstAddr += 4;
    }

    if (resWord > 0) {
      // TODO : when moving to clang, remove following line and pass mask as extra last parameter
      __asm__ __volatile__("mov.m.x m0, %[mask], 0x0 \n" : : [ mask ] "r"(mask) :);
      op.doOpVectScatter(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset);
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
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

  ////////////////////////////////////////////////////////////////////////////////
  // individual functions per operator (forwarding call to the previous ones with
  // the proper parameters)
  ////////////////////////////////////////////////////////////////////////////////
#define EltWiseInst(name, opType)                                                                                      \
  template <ElemKind src1ElK, ElemKind src2ElK>                                                                        \
  INLINE_ATTR void fwdLib##name##Inst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,               \
                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {           \
    inlining::fwdLibElementBoolInst<src1ElK, src2ElK, opType>(outT, in1T, in2T, flags, minionOffset, assignedMinions); \
  }                                                                                                                    \
  template <ElemKind src1ElK, ElemKind src2ElK>                                                                        \
  INLINE_ATTR void fwdLib##name##InstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,       \
                                              const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {   \
    inlining::fwdLibElementBoolInstThreaded<src1ElK, src2ElK, opType>(outT, in1T, in2T, flags, minionOffset,           \
                                                                      assignedMinions);                                \
  }                                                                                                                    \
  template <ElemKind src1ElK, ElemKind src2ElK>                                                                        \
  INLINE_ATTR void fwdLib##name##InstVectorized(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,     \
                                                const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) { \
    inlining::fwdLibElementBoolInstVectorized<src1ElK, src2ElK, opType>(outT, in1T, in2T, flags, minionOffset,         \
                                                                        assignedMinions);                              \
  }

EltWiseInst(ElementCmpEQ, CmpEQ) EltWiseInst(ElementCmpNEQ, CmpNEQ) EltWiseInst(ElementCmpLT, CmpLT)
  EltWiseInst(ElementCmpLTE, CmpLTE)

#undef EltWiseInst

#define LogicalInst(name, opType)                                                                                      \
  INLINE_ATTR void fwdLib##name##Inst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,               \
                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {           \
    inlining::fwdLibElementBoolInst<BoolTy, BoolTy, opType>(outT, in1T, in2T, flags, minionOffset, assignedMinions);   \
  }                                                                                                                    \
  INLINE_ATTR void fwdLib##name##InstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,       \
                                              const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {   \
    inlining::fwdLibElementBoolInstThreaded<BoolTy, BoolTy, opType>(outT, in1T, in2T, flags, minionOffset,             \
                                                                    assignedMinions);                                  \
  }

    LogicalInst(ElementAnd, ElementAnd) LogicalInst(ElementOr, ElementOr) LogicalInst(ElementXor, ElementXor)

#undef LogicalInst

} // namespace inlining

} // namespace dnn_lib

#endif // _ELEMENT_BOOL_INST_H_

