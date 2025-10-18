/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_
#define _SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_

#include "LibCommon.h"
#include "LibTensor.h"
#include "LibTypes.h"
#include "utils.h"
#include <assert.h>
#include <limits>

namespace dnn_lib {

namespace inlining {

template <ElemKind elKind, ElemKind idxKind>
INLINE_ATTR typename std::enable_if_t<(isQuantizedElemKind(elKind) || (elKind == Float16Ty)), void>

fwdLibSparseLengthsWeightedSumInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
                                   uint64_t flags, const uint32_t minionOffset = 0,
                                   [[maybe_unused]] const uint32_t assignedMinions = 0) {
  if (get_minion_id() != minionOffset) return;

  assert(in1T->getElementType() == outT->getElementType());
  assert((in3T->getElementType() == Int64ITy) || (in3T->getElementType() == Int32ITy));
  assert(in4T->getElementType() == Int32ITy);

  using elkType = typename elemKind2elemTy<elKind>::type; 
  using idxType = typename elemKind2elemTy<idxKind>::type; 

  auto outH = outT->getHandle<elkType>();
  auto dataH = in1T->getHandle<elkType>();
  auto weightH = in2T->getHandle<elkType>();
  auto idxH = in3T->getHandle<idxType>();
  auto lengthH = in4T->getHandle<int32_t>();

  outH.zero();

  size_t segments = in4T->dims()[0];

  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += lengthH.raw(i);
  }

  assert(totalLength <= in3T->dims()[0]);

  const size_t lineSize = in1T->size() / in1T->dims()[0];

  auto weight = weightH.begin();
  auto idx = idxH.begin();
  auto out = outH.begin();
  auto len = lengthH.begin();
  dim_array_t inCoords = {0};

  for (size_t i = 0; i < segments; i++, out.step(0), ++len) {
    for ( int32_t j = 0; j < *len; ++j, ++weight, ++idx) {
      inCoords[0] = *idx;
      auto dataIn = dataH.getIterator(inCoords);
      auto dataOut = out;
      for (dim_t k = 0; k < lineSize; k++) {
        float wei = 1.0;
        float dtin = 0.0;
        float accumDtOut = 0.0;
        float tmp = 0.0;
        if (elKind == Float16Ty) {
          float dst = 0.0;
          convertFp16ToFp32(static_cast<uint16_t>(*dataIn), dst);
          dtin = dst;
          if (in2T != nullptr) {
            convertFp16ToFp32(static_cast<uint16_t>(*weight), dst);
            wei = dst;          
          } 
          convertFp16ToFp32(static_cast<uint16_t>(*dataOut), dst);
          accumDtOut = dst;                         
        }
        else {
    if (in2T != nullptr) {
      wei = dequantize<elkType>((*weight), in2T->getScale(), in2T->getOffset());
    }
          dtin = dequantize<elkType>((*dataIn), in1T->getScale(), in1T->getOffset());
          accumDtOut = dequantize<elkType>((*dataOut), outT->getScale(), outT->getOffset());
        }
        tmp = dtin * wei;
        if (elKind == Float16Ty) {
          uint16_t dst = 0;
          accumDtOut += tmp;
          convertFp32ToFp16(accumDtOut, dst);
          *dataOut = dst;
        }
        else {
          accumDtOut += tmp;
          *dataOut = quantize<elkType>(accumDtOut, outT->getScale(), outT->getOffset());
        }
        ++dataIn;
        ++dataOut;
      }
    }
  }

  outT->evict(DO_EVICTS);
}

template <ElemKind elKind, ElemKind idxKind>
INLINE_ATTR
  typename std::enable_if_t<(!isQuantizedElemKind(elKind) && (elKind != Float16Ty) && (elKind != BoolTy)), void>
  fwdLibSparseLengthsWeightedSumInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                                     LibTensor* in4T, uint64_t flags, const uint32_t minionOffset = 0,
                                     [[maybe_unused]] const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(in1T->getElementType() == outT->getElementType());
  assert((in3T->getElementType() == Int64ITy) || (in3T->getElementType() == Int32ITy));
  assert(in4T->getElementType() == Int32ITy);

  using elkType = typename elemKind2elemTy<elKind>::type; 
  using idxType = typename elemKind2elemTy<idxKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = in1T->getHandle<elkType>();
  auto weightH = in2T->getHandle<elkType>();
  auto idxH = in3T->getHandle<idxType>();
  auto lengthH = in4T->getHandle<int32_t>();

  outH.zero();
  
  size_t segments = in4T->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += lengthH.raw(i);
  }

  assert(totalLength <= in4T->dims()[0]);

  size_t lineSize = in1T->size() / in1T->dims()[0];
  auto weight = weightH.begin();
  auto idx = idxH.begin();
  auto out = outH.begin();
  auto len = lengthH.begin();
  dim_array_t inCoords = {0};

  for (size_t i = 0; i < segments; i++, out.step(0), ++len) {
    for ( int32_t j = 0; j < *len; ++j, ++weight, ++idx){
      inCoords[0] = *idx;
      auto dataIn = dataH.getIterator(inCoords);
      auto dataOut = out;
      for (dim_t k = 0; k < lineSize; k++) {
        *dataOut += (*dataIn) * (*weight);
        ++dataIn;
        ++dataOut;
      }
    }
  }
  
  outT->evict(DO_EVICTS);
}

template <ElemKind elKind, ElemKind idxKind>
INLINE_ATTR typename std::enable_if_t<(isQuantizedElemKind(elKind) || (elKind == Float16Ty)), void>
fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                                           LibTensor* in4T, uint64_t flags, const uint32_t minionOffset = 0,
                                           const uint32_t assignedMinions = 0) {
  (void)outT;
  (void)in1T;
  (void)in2T;
  (void)in3T;
  (void)in4T;
  (void)flags;
  (void)minionOffset;
  (void)assignedMinions;

  // Previous threaded version was not working properly and was deactivated, check it at:
  // https://gitlab.esperanto.ai/software/dnn-library/blob/6d3bd3be6d59e70068190b011de9662ab0fe03d1/include/inlining/SparseLengthsWeightedSumInst.h
  // TODO: new threaded version based on SparseLengthsSumInst.h threaded version
}

template <ElemKind elKind, ElemKind idxKind>
INLINE_ATTR
  typename std::enable_if_t<(!isQuantizedElemKind(elKind) && (elKind != Float16Ty) && (elKind != BoolTy)), void>
  fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                                             LibTensor* in4T, uint64_t flags, const uint32_t minionOffset = 0,
                                             const uint32_t assignedMinions = 0) {

  (void)outT;
  (void)in1T;
  (void)in2T;
  (void)in3T;
  (void)in4T;
  (void)flags;
  (void)minionOffset;
  (void)assignedMinions;

  // Previous threaded version was not working properly and was deactivated, check it at:
  // https://gitlab.esperanto.ai/software/dnn-library/blob/6d3bd3be6d59e70068190b011de9662ab0fe03d1/include/inlining/SparseLengthsWeightedSumInst.h
  // TODO: new threaded version based on SparseLengthsSumInst.h threaded version
}

} // namespace dnn_lib

} // namespace inlining

#endif // _SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_
