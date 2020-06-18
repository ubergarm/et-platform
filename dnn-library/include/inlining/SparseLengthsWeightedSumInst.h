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

  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    float accum[lineSize] = {0.0};
    for (int32_t j = 0; j < lengthH.at(std::array<size_t,1>{i}); j++) {
      float weight = 0.0;
      if (elKind == Float16Ty) {
	float dst = 0.0;
	convertFp16ToFp32(static_cast<uint16_t>(weightH.at(std::array<size_t,1>{curIdx})), dst);
	weight = dst;
      }
      else
	weight = dequantize<elkType>(weightH.at(std::array<size_t,1>{curIdx}), 
				     in2T->getScale(), in2T->getOffset());
      size_t offsetIn = idxH.at(std::array<size_t,1>{curIdx}) * lineSize;      
      for (dim_t k = 0; k < lineSize; k++) {
	if (elKind == Float16Ty) {
	  float dst = 0.0;
	  convertFp16ToFp32(static_cast<uint16_t>(dataH.at(std::array<size_t,1>{offsetIn++})), dst);
	  accum[k] += weight * dst;
	}
	else{
	  accum[k] += weight * dequantize<elkType>(dataH.at(std::array<size_t,1>{offsetIn++}), 
						   in1T->getScale(), in1T->getOffset());
	}
      }
      curIdx++;
    }      
    size_t offsetOut = i * lineSize;
    for (dim_t k = 0; k < lineSize; k++){
      if (elKind == Float16Ty) {
	uint16_t dst = 0;
	convertFp32ToFp16(accum[k], dst);
	outH.at(std::array<size_t,1>{offsetOut++}) = dst;
      }
      else {
	outH.at(std::array<size_t,1>{offsetOut++}) = quantize<elkType>(accum[k], outT->getScale(), outT->getOffset());
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
  
  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    for (size_t j = 0; j < lengthH.at(std::array<size_t,1>{i}); j++) {
      elkType weight = weightH.at(std::array<size_t,1>{curIdx});
      size_t offsetIn = idxH.at(std::array<size_t,1>{curIdx++}) * lineSize;
      size_t offsetOut = i * lineSize;
      for (dim_t k = 0; k < lineSize; k++) 
  	outH.at(std::array<size_t,1>{offsetOut++}) += 
	  dataH.at(std::array<size_t,1>{offsetIn++}) * weight;
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

  outT->evict(DO_EVICTS, first, count);
}


template <ElemKind elKind, ElemKind idxKind>
inline typename std::enable_if_t<(!isQuantizedElemKind(elKind) && 
				  (elKind != Float16Ty) && (elKind != BoolTy)), void>
fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                           LibTensor* in3T, LibTensor* in4T, uint64_t flags,
                                           const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {) {

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

  size_t lineSize = in1T->size() / in1T->dims()[0];

  auto out = outT->getHandle<elkType>().getIterator(first);
  auto in = in1T->getHandle<elkType>().getIterator(out.coords());
  
  dim_array_t coord = outT->offset2Coord(first);
  
  for(; out.offset() < first + count;++out, ++in) {
    size_t curIdx = 0;
    size_t segment_begin = ranges[coord[0]];
    size_t segment_end = segment_begin + lengthH.at(std::array<size_t,1>{coord[0]});

    float res = 0;
    for (size_t k = segment_begin; k < segment_end; k++) {
      size_t aux = idxH.at(std::array<size_t,1>{k});
      aux += (in.offset()  * weightH.at(std::array<size_t,1>{k}));      
      res += dataH.at(std::array<size_t,1>{aux});
    }
    outH.at(std::array<size_t,1>{out.offset()}) = res;
  }

  outT->evict(DO_EVICTS, first, count);
}

/* // This version does NOT support Tensors of more than 2 dimensions with padding */
/* template <ElemKind elKind> */
/* inline void fwdLibSparseLengthsWeightedSumInst(LibTensor* outT, LibTensor* in1T, */
/*                                                LibTensor* in2T, LibTensor* in3T, */
/*                                                LibTensor* in4T, */
/* 					       uint64_t flags) { */
/*   unsigned int minionId = get_minion_id(); */
/*   if (minionId != 0) */
/*     return; */

/*   using elkType = typename elemKind2elemTy<elKind>::type; */

/*   /\* maintain compatibility through the new Iface Libtensor *\/ */
/*   void* dst = outT->getRawDataPointer<void>(); */
/*   void* data = in1T->getRawDataPointer<void>(); */
/*   void* weight = in2T->getRawDataPointer<void>(); */
  
/*   // Addresser<srcType> tOutput(pdst, scale[4], offset[4]); */
/*   Addresser<elkType> tOutput(dst, outT->getScale(), outT->getOffset()); */
/*   // const Addresser<srcType> tAInput(pdata, scale[0], offset[0]); */
/*   const Addresser<elkType> tAInput(data, in1T->getScale(), in1T->getOffset()); */
/*   // const Addresser<srcType> tWInput(pweights, scale[1], offset[1]); */
/*   const Addresser<elkType> tWInput(weight, in2T->getScale(), in2T->getOffset()); */
/*   // long long *indices = (long long *)pindices; */
/*   long long *indices = in3T->getRawDataPointer<long long>(); */
/*   // int32_t *lengths = (int32_t *)plengths; */
/*   int32_t *lengths = in4T->getRawDataPointer<int32_t>(); */

/*   // unsigned int *dataIndex = (unsigned int *)pdataDims; */
/*   const dim_t *dataIndex = in1T->dims().data(); */
/*   // unsigned int *dstPitch = (unsigned int *)pdstPitches; */
/*   const dim_t *dstPitch = outT->strides().data(); */
/*   // unsigned int *dataPitch = (unsigned int *)pdataPitches; */
/*   const dim_t *dataPitch = in1T->strides().data(); */
/*   // unsigned int *weightPitch = (unsigned int *)pweightsPitches; */
/*   const dim_t *weightPitch = in2T->strides().data(); */

/*   unsigned int pdstDimNum = static_cast<unsigned int>(outT->ndims()); */

/*   size_t segments = in3T->dims()[0]; //pLengthsSize; */
/*   size_t totalLength = 0; */
/*   for (size_t i = 0; i < segments; i++) { */
/*     totalLength += lengths[i]; */
/*   } */

/*   size_t totalSize = 1; */
/*   for (size_t i = 0; i < pdstDimNum; i++) { */
/*     totalSize *= dataIndex[i]; */
/*   } */
/*   size_t lineSize = totalSize / dataIndex[0]; */

/*   // Output tensor should be zero at the begin */
/*   size_t curIdx = 0; */
/*   for (size_t i = 0; i < segments; i++) { */
/*     // NOTE : Not C++ compliant?  Fails with clang. */
/*     // float tmp[lineSize] = { 0.0f }; */
/*     float tmp[lineSize]; */
/*     for (size_t j = 0; j < lineSize; j++) */
/*       tmp[j] = 0.0f; */
/*     for (size_t j = 0, e = lengths[i]; j < e; j++) { */
/*       float weight = tWInput[curIdx * weightPitch[0]]; */
/*       size_t offsetIn = indices[curIdx] * dataPitch[0]; */
/*       for (size_t k = 0; k < lineSize; k++) { */
/*         tmp[k] += tAInput[offsetIn] * weight; */
/*         offsetIn++; */
/*       } */
/*       curIdx++; */
/*     } */
/*     size_t offsetOut = i * dstPitch[0]; */
/*     for (size_t k = 0; k < lineSize; k++) { */
/*       tOutput[offsetOut] = tmp[k]; */
/*       offsetOut++; */
/*     } */
/*   } */
/*   outT->evict(DO_EVICTS); */
/* } */

/* // This version DOES support Tensors of more than 2 dimensions with padding */
/* template <ElemKind elKind> */
/* inline void fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* outT, */
/*                                                        LibTensor* in1T, */
/*                                                        LibTensor* in2T, */
/*                                                        LibTensor* in3T, */
/*                                                        LibTensor* in4T, */
/*                                                        uint64_t flags) { */

/*   unsigned int minionId = get_minion_id(); */
/*   unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES; */
/*   if (minionId >= activeMinions) */
/*     return; */

/*   using elkType = typename elemKind2elemTy<elKind>::type; */
/*   /\* maintain compatibility through the new Iface Libtensor *\/ */
/*   void* dst = outT->getRawDataPointer<void>(); */
/*   void* data = in1T->getRawDataPointer<void>(); */
/*   void* weight = in2T->getRawDataPointer<void>(); */
 
/*   // Addresser<srcType> tOutput(pdst, scale[4], offset[4]); */
/*   Addresser<elkType> tOutput(dst, outT->getScale(), outT->getOffset()); */
/*   // const Addresser<srcType> tAInput(pdata, scale[0], offset[0]); */
/*   const Addresser<elkType> tAInput(data, in1T->getScale(), in2T->getOffset()); */
/*   // const Addresser<srcType> tWInput(pweights, scale[1], offset[1]); */
/*   const Addresser<elkType> tWInput(weight, in2T->getScale(), in2T->getOffset()); */
/*   // long long *indices = (long long *)pindices; */
/*   long long *indices = in3T->getRawDataPointer<long long>(); */
/*   // int32_t *lengths = (int32_t *)plengths; */
/*   int32_t *lengths = in4T->getRawDataPointer<int32_t>(); */

/*   // unsigned int *dstIndex = (unsigned int *)pdstDims; */
/*   const dim_t *dstIndex = outT->dims().data(); */
/*   // unsigned int *dstPitch = (unsigned int *)pdstPitches; */
/*   const dim_t *dstPitch = outT->strides().data(); */
/*   // unsigned int *dataPitch = (unsigned int *)pdataPitches; */
/*   const dim_t *dataPitch = in1T->strides().data(); */
/*   // unsigned int *weightPitch = (unsigned int *)pweightsPitches; */
/*   const dim_t *weightPitch = in2T->strides().data(); */

/*   unsigned int pdstDimNum = static_cast<unsigned int>(outT->ndims()); */
  
/*   unsigned int numElemsDst = dstPitch[0] * dstIndex[0]; */
/*   unsigned int initialAddr, maxRead; */
/*   size_t typeSize = getsize<elkType>(); */
/*   getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, */
/*                         minionId, activeMinions); */
/*   if (maxRead == 0) */
/*     return; */

/*   size_t segments = in3T->dims()[0];  //pLengthsSize; */
/*   size_t ranges[segments]; */
/*   size_t totalLength = 0; */
/*   for (size_t i = 0; i < segments; i++) { */
/*     ranges[i] = totalLength; */
/*     totalLength += lengths[i]; */
/*   } */

/*   unsigned int coord[pdstDimNum]; */
/*   unsigned int k = 0; */
/*   getNonPaddingCoordinates(coord, initialAddr, pdstDimNum, dstPitch, dstIndex, */
/*                            k); */

/*   unsigned int offsetOut = 0; */
/*   for (unsigned int i = 0; i < k; i++) */
/*     offsetOut += coord[i] * dstPitch[i]; */
/*   if (offsetOut >= numElemsDst) */
/*     return; */

/*   unsigned int posMax = initialAddr + maxRead; */
/*   bool done = false; */
/*   while (!done && (offsetOut < posMax)) { */
/*     size_t segment_begin = ranges[coord[0]]; */
/*     size_t segment_end = segment_begin + lengths[coord[0]]; */

/*     size_t offsetIn = 0; */
/*     for (unsigned int i = 1; i < pdstDimNum; i++) */
/*       offsetIn += coord[i] * dataPitch[i]; */

/*     float res = 0; */
/*     for (size_t k = segment_begin; k < segment_end; k++) { */
/*       res += tAInput[indices[k] * dataPitch[0] + offsetIn] * */
/*              (float)tWInput[k * weightPitch[0]]; */
/*     } */

/*     tOutput[offsetOut] = res; */
/*     done = getOffsets(pdstDimNum, coord, offsetOut, dstIndex, dstPitch); */
/*   } */
/*   if (!DO_EVICTS) */
/*     return; */
/*   unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES; */
/*   if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion); */
/* } */

} // namespace dnn_lib

} // namespace inlining

#endif // _SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_
