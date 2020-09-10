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

#ifndef _SPARSE_LENGTHS_SUM_INST_H
#define _SPARSE_LENGTHS_SUM_INST_H

#include "LibTypes.h"

#include "SparseLengthsWeightedSumInst.h"

namespace dnn_lib {
namespace inlining {

/* @brief Pulls in slices of the input tensor, groups them into segments 
 * and applies 'Sum' to each segment. Segments are defined by their LENGTHS. 
 * This op is basically Gather and LengthsSum fused together. 
 * INDICES should contain integers in range 0..N-1 where N is the first dimension of DATA. 
 * INDICES represent which slices of DATA need to be pulled in. 
 * LENGTHS is a vector that defines slice sizes by first dimention of DATA. 
 * Values belonging to the same segment are aggregated together. sum(LENGTHS) has to
 * match INDICES size. The first dimension of the output is equal to the number of 
 * input segment, i.e. len(LENGTHS) . Other dimensions are inherited from the input 
 * tensor. Summation is done element-wise across slices of the input tensor and doesn?t
 *  change the shape of the individual blocks
 */
template <ElemKind elKind, ElemKind idxKind>
  inline typename std::enable_if_t<(isQuantizedElemKind(elKind) || elKind == Float16Ty), void>
fwdLibSparseLengthsSumInst(LibTensor* outT, LibTensor* data, LibTensor* indices,
        LibTensor* length, uint64_t flags, 
        const uint32_t minionOffset = 0, 
        const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(data->getElementType() == outT->getElementType());
  assert((indices->getElementType() == Int64ITy) || (indices->getElementType() == Int32ITy));
  assert(length->getElementType() == Int32ITy);

  using elkType = typename elemKind2elemTy<elKind>::type; 
  using idxType = typename elemKind2elemTy<idxKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = data->getHandle<elkType>();
  auto idxH = indices->getHandle<idxType>();
  auto lengthH = length->getHandle<int32_t>();

  outH.zero();


  size_t segments = length->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += lengthH.raw(i);
  }

  assert(totalLength <= indices->dims()[0]);

  size_t lineSize = data->size() / data->dims()[0];
  auto idx = idxH.begin();
  auto out = outH.begin();
  auto len = lengthH.begin();
  dim_array_t inCoords = {0};

  for (size_t i = 0; i < segments; i++, out.step(0), ++len) {
    for ( int32_t j = 0; j < *len; ++j, ++idx){
      inCoords[0] = *idx;
      auto dataIn = dataH.getIterator(inCoords);
      auto dataOut = out;
      for (dim_t k = 0; k < lineSize; k++) {
  float dst = 0.0;
  float aux = 0.0;
  if (elKind == Float16Ty) {
    uint16_t tmp = 0;
    convertFp16ToFp32(static_cast<uint16_t>(*dataIn), dst);
    convertFp16ToFp32(static_cast<uint16_t>(*dataOut), aux);
    aux += dst;
    convertFp32ToFp16(aux, tmp);
    (*dataOut) = tmp;
  }
  else {
    dst = dequantize<elkType>(*dataIn, data->getScale(), data->getOffset());
    aux = dequantize<elkType>(*dataOut, outT->getScale(), outT->getOffset());
    aux += dst;
    (*dataOut) = quantize<elkType>(aux, outT->getScale(), outT->getOffset());
  }
        ++dataIn;
        ++dataOut;
      }
    }
  }

  outT->evict(DO_EVICTS);

}


template <ElemKind elKind, ElemKind idxKind>
inline typename std::enable_if_t<(!isQuantizedElemKind(elKind) && 
                                  (elKind != Float16Ty) && (elKind != BoolTy)), void>
fwdLibSparseLengthsSumInst(LibTensor* outT, LibTensor* data, 
        LibTensor* indices, LibTensor* length, 
        uint64_t flags, const uint32_t minionOffset = 0, 
        const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(data->getElementType() == outT->getElementType());
  assert((indices->getElementType() == Int64ITy) || (indices->getElementType() == Int32ITy));
  assert(length->getElementType() == Int32ITy);

  using elkType = typename elemKind2elemTy<elKind>::type; 
  using idxType = typename elemKind2elemTy<idxKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = data->getHandle<elkType>();
  auto idxH = indices->getHandle<idxType>();
  auto lengthH = length->getHandle<int32_t>();

  outH.zero();

  size_t segments = length->dims()[0];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += lengthH.raw(i);
  }

  assert(totalLength <= indices->dims()[0]);

  size_t lineSize = data->size() / data->dims()[0];
  auto idx = idxH.begin();
  auto out = outH.begin();
  auto len = lengthH.begin();
  dim_array_t inCoords = {0};

  for (size_t i = 0; i < segments; i++, out.step(0), ++len) {
    for ( int32_t j = 0; j < *len; ++j, ++idx){
      inCoords[0] = *idx;
      auto dataIn = dataH.getIterator(inCoords);
      auto dataOut = out;
      for (dim_t k = 0; k < lineSize; k++) {
        *dataOut += (*dataIn);
        ++dataIn;
        ++dataOut;
      }
    }
  }

  outT->evict(DO_EVICTS);
}

} // inlining
} // namespace

#endif
