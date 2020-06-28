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

#ifndef _SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_
#define _SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_

#include <limits>
#include <assert.h>

#include "utils.h"
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind elKind, ElemKind idxKind>
inline typename std::enable_if_t<(isQuantizedElemKind(elKind) || (elKind==Float16Ty)), void>
fwdLibSparseLengthsWeightedSumInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                   LibTensor* in3T, LibTensor* in4T, 
                                   uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
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
inline typename std::enable_if_t<(!isQuantizedElemKind(elKind) && 
                                  (elKind != Float16Ty) && (elKind != BoolTy)), void>
fwdLibSparseLengthsWeightedSumInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                   LibTensor* in3T, LibTensor* in4T,
                                   uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

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
inline typename std::enable_if_t<(isQuantizedElemKind(elKind) || (elKind == Float16Ty)), void>
fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                           LibTensor* in3T, LibTensor* in4T, uint64_t flags,
                                           const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;


  ////////////////////////////////////////////////////////////////////////////////
  // partition work between minions in multiples of CL
  ////////////////////////////////////////////////////////////////////////////////  
  size_t first; // first element in raw array to process
  size_t count; // nr  elements in to process (will be in multiples of CL)

  outT->partitionCL(minionId, activeMinions, first, count);

  if (unlikely(count == 0)) return; // minion has no work to do

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
  size_t ranges[segments];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    ranges[i] = totalLength;
    totalLength += lengthH.raw(i);
  }

  assert(totalLength <= in4T->dims()[0]);


  auto out = outT->getHandle<elkType>().getIterator(first);
  auto in = in1T->getHandle<elkType>().getIterator(out.coords());
  
  dim_array_t coord = outT->offset2Coord(first);
  
  for(; out.offset() < first + count;++out, ++in) {
    size_t segment_begin = ranges[coord[0]];
    size_t segment_end = segment_begin + lengthH.at(std::array<size_t,1>{coord[0]});

    float res = 0;
    for (size_t k = segment_begin; k < segment_end; k++) {

      size_t aux = idxH.at(std::array<size_t,1>{k});
      float wei = 1.0;
      float res = 0.0;
      if(elKind == Float16Ty) {	
	float dst = 0.0;
	if (in2T != nullptr) {
	  convertFp16ToFp32(static_cast<uint16_t>(weightH.at(std::array<size_t,1>{k})), dst);
	  wei = dst;
	}
	convertFp16ToFp32(static_cast<uint16_t>(in.offset()), dst);
	aux += dst * wei;
	convertFp16ToFp32(static_cast<uint16_t>(dataH.at(std::array<size_t,1>{k})), dst);
	res += dst;
      }
      else {
	float inoffset = 0.0;
	if (in2T != nullptr) {
	  wei = dequantize<elkType>(weightH.at(std::array<size_t,1>{k}), in2T->getScale(), in2T->getOffset());
	}
	inoffset = dequantize<elkType>(in.offset(), in1T->getScale(), in1T->getOffset());
	aux += inoffset * wei;
	res += dequantize<elkType>(dataH.at(std::array<size_t,1>{aux}), in1T->getScale(), in1T->getOffset()); 
      }
    }

    if (elKind == Float16Ty) {      
      uint16_t dst = 0;
      convertFp32ToFp16(res, dst);
      outH.at(std::array<size_t,1>{out.offset()}) = dst;
    }
    else {
      outH.at(std::array<size_t,1>{out.offset()}) = quantize<elkType>(res, outT->getScale(), outT->getOffset());
    }
  }

  outT->evict(DO_EVICTS, first, count);
}


template <ElemKind elKind, ElemKind idxKind>
inline typename std::enable_if_t<(!isQuantizedElemKind(elKind) && 
                                  (elKind != Float16Ty) && (elKind != BoolTy)), void>
fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                           LibTensor* in3T, LibTensor* in4T, uint64_t flags,
                                           const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  ////////////////////////////////////////////////////////////////////////////////
  // partition work between minions in multiples of CL
  ////////////////////////////////////////////////////////////////////////////////  
  size_t first; // first element in raw array to process
  size_t count; // nr  elements in to process (will be in multiples of CL)

  outT->partitionCL(minionId, activeMinions, first, count);

  if (unlikely(count == 0)) return; // minion has no work to do

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
  size_t ranges[segments];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    ranges[i] = totalLength;
    totalLength += lengthH.raw(i);
  }

  assert(totalLength <= in4T->dims()[0]);


  auto out = outT->getHandle<elkType>().getIterator(first);
  auto in = in1T->getHandle<elkType>().getIterator(out.coords());
  
  dim_array_t coord = outT->offset2Coord(first);
  
  for(; out.offset() < first + count;++out, ++in) {
    size_t segment_begin = ranges[coord[0]];
    size_t segment_end = segment_begin + lengthH.at(std::array<size_t,1>{coord[0]});

    float res = 0;
    for (size_t k = segment_begin; k < segment_end; k++) {
      size_t aux = idxH.at(std::array<size_t,1>{k});
      if (in2T != nullptr) {
	aux += (in.offset()  * weightH.at(std::array<size_t,1>{k}));      
      }
      else {
	aux += in.offset();
      }
      res += dataH.at(std::array<size_t,1>{aux});
    }
    outH.at(std::array<size_t,1>{out.offset()}) = res;
  }

  outT->evict(DO_EVICTS, first, count);
}

} // namespace dnn_lib

} // namespace inlining

#endif // _SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_
