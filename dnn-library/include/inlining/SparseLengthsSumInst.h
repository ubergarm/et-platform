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
INLINE_ATTR typename std::enable_if_t<(isQuantizedElemKind(elKind) || elKind == Float16Ty), void>
fwdLibSparseLengthsSumInst(LibTensor* outT, LibTensor* data, LibTensor* indices, LibTensor* length, uint64_t flags,
                           const uint32_t minionOffset = 0, [[maybe_unused]] const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(data->getElementType() == outT->getElementType());
  assert((indices->getElementType() == Int64ITy) || (indices->getElementType() == Int32ITy));
  assert(length->getElementType() == Int32ITy);

  using elkType = typename elemKind2elemTy<elKind>::type; 
  using idxType = typename elemKind2elemTy<idxKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = data->getHandle<elkType>();
  auto idxH = indices->getHandle<idxType>();
  auto lengthH = length->getHandle<uint32_t>();

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
    for (size_t j = 0; j < *len; ++j, ++idx) {
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
        } else {
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
INLINE_ATTR
  typename std::enable_if_t<(!isQuantizedElemKind(elKind) && (elKind != Float16Ty) && (elKind != BoolTy)), void>
  fwdLibSparseLengthsSumInst(LibTensor* outT, LibTensor* data, LibTensor* indices, LibTensor* length, uint64_t flags,
                             const uint32_t minionOffset = 0, [[maybe_unused]] const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(data->getElementType() == outT->getElementType());
  assert((indices->getElementType() == Int64ITy) || (indices->getElementType() == Int32ITy));
  assert(length->getElementType() == Int32ITy);

  using elkType = typename elemKind2elemTy<elKind>::type; 
  using idxType = typename elemKind2elemTy<idxKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = data->getHandle<elkType>();
  auto idxH = indices->getHandle<idxType>();
  auto lengthH = length->getHandle<uint32_t>();

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
    for (size_t j = 0; j < *len; ++j, ++idx) {
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

template <ElemKind elKind, ElemKind idxKind>
INLINE_ATTR typename std::enable_if_t<(isQuantizedElemKind(elKind) || (elKind == Float16Ty)), void>
fwdLibSparseLengthsSumInstThreaded(LibTensor* outT, LibTensor* data, LibTensor* indices, LibTensor* lengths,
                                   uint64_t flags, const uint32_t minionOffset = 0,
                                   const uint32_t assignedMinions = 0) {

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  assert(data->getElementType() == outT->getElementType() && "Input and Ouput tensor datatypes must the same");
  assert(((indices->getElementType() == Int64ITy) || (indices->getElementType() == Int32ITy)) &&
         "indices datatype must be integer 32/64-bit");
  assert(lengths->getElementType() == Int32ITy && "Lenghts datatype must be 32-bit integer");

  // Obtain data partition
  size_t offset;  // first element in raw array to process
  size_t maxRead; // num elements to process (will be in multiples of CL)
  outT->partitionCL(minionId, activeMinions, offset, maxRead);
  if (unlikely(maxRead == 0)) {
    return; // minion has no work to do
  }

  using elkType = typename elemKind2elemTy<elKind>::type;
  using idxType = typename elemKind2elemTy<idxKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = data->getHandle<elkType>();
  auto idxH = indices->getHandle<idxType>();
  auto lengthH = lengths->getHandle<uint32_t>();

  const dim_t segmentStride = outT->strides()[0];
  const dim_t segments = lengths->dims()[0];
  const dim_t numColumns = data->dims()[1];

  // Post: segmentPtr contains a pointer to the first element on each segment.
  // This computation is redundant and could be optimized so segmentPtr[]
  // is filled by one minion then shared.
  size_t segmentPtr[segments + 1];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    segmentPtr[i] = totalLength;
    totalLength += lengthH.raw(i);
  }
  segmentPtr[segments] = totalLength;
  assert(totalLength <= lengths->dims()[0] && "sum(Lengths) must be equal to len(indices)");

  std::fill(outH.getIterator(offset), outH.getIterator(offset + maxRead), 0);

  // Obtain the boundaries of the segments/elements proceesed by this minion.
  auto firstSegment = offset / segmentStride;
  auto lastSegment = (offset + maxRead - 1) / segmentStride;
  // Some minion offsets might fall halfway of a segment.
  auto firstSegmentBegin = offset % segmentStride; // Index of the first element to process
  auto lastSegmentEnd =
    std::min(numColumns - 1, (offset + maxRead - 1) % segmentStride); // Index of the last element to process

  auto idx = idxH.getIterator(segmentPtr[firstSegment]);
  auto out = outH.getIterator(offset);
  auto len = lengthH.getIterator(firstSegment);
  dim_array_t dataCoords = {0};

  for (size_t s = firstSegment; s <= lastSegment; s++, out.step(0), ++len) {
    size_t startElem = (s == firstSegment) ? firstSegmentBegin : 0;
    size_t lastElem = (s == lastSegment) ? lastSegmentEnd + 1 : numColumns;
    for (size_t j = 0; j < *len; j++, ++idx) {
      auto outIt = out;
      dataCoords[0] = *idx;
      auto dataIt = dataH.getIterator(dataCoords); // Iterator to the data row to accumulate
      dataIt += static_cast<std::ptrdiff_t>(startElem);
      for (size_t k = startElem; k < lastElem; k++) {
        float dataValue = 0.0;
        float accumValue = 0.0;
        if constexpr (elKind == Float16Ty) {
          uint16_t accumValueFP16 = 0;
          convertFp16ToFp32(static_cast<uint16_t>(*dataIt), dataValue);
          convertFp16ToFp32(static_cast<uint16_t>(*outIt), accumValue);
          accumValue += dataValue;
          convertFp32ToFp16(accumValue, accumValueFP16);
          (*outIt) = accumValueFP16;
        } else {
          dataValue = dequantize<elkType>(*dataIt, data->getScale(), data->getOffset());
          accumValue = dequantize<elkType>(*outIt, outT->getScale(), outT->getOffset());
          accumValue += dataValue;
          (*outIt) = quantize<elkType>(accumValue, outT->getScale(), outT->getOffset());
        }
        ++dataIt;
        ++outIt;
      }
    }
  }
  outT->evict(DO_EVICTS, offset, maxRead);
}

template <ElemKind elKind, ElemKind idxKind>
INLINE_ATTR
  typename std::enable_if_t<(!isQuantizedElemKind(elKind) && (elKind != Float16Ty) && (elKind != BoolTy)), void>
  fwdLibSparseLengthsSumInstThreaded(LibTensor* outT, LibTensor* data, LibTensor* indices, LibTensor* lengths,
                                     uint64_t flags, const uint32_t minionOffset = 0,
                                     const uint32_t assignedMinions = 0) {

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  assert(data->getElementType() == outT->getElementType() && "Input and Ouput tensor datatypes must the same");
  assert(((indices->getElementType() == Int64ITy) || (indices->getElementType() == Int32ITy)) &&
         "indices datatype must be integer 32/64-bit");
  assert(lengths->getElementType() == Int32ITy && "Lenghts datatype must be 32-bit integer");

  // Obtain data partition
  size_t offset;  // first element in raw array to process
  size_t maxRead; // num elements to process (will be in multiples of CL)
  outT->partitionCL(minionId, activeMinions, offset, maxRead);
  if (unlikely(maxRead == 0)) {
    return; // minion has no work to do
  }

  using elkType = typename elemKind2elemTy<elKind>::type;
  using idxType = typename elemKind2elemTy<idxKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = data->getHandle<elkType>();
  auto idxH = indices->getHandle<idxType>();
  auto lengthH = lengths->getHandle<int32_t>();

  const dim_t segmentStride = outT->strides()[0];
  const dim_t segments = lengths->dims()[0];
  const dim_t numColumns = data->dims()[1];

  // Post: segmentPtr contains a pointer to the first element on each segment.
  // This computation is redundant and could be optimized so segmentPtr[]
  // is filled by one minion then shared.
  size_t segmentPtr[segments + 1];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    segmentPtr[i] = totalLength;
    totalLength += lengthH.raw(i);
  }
  segmentPtr[segments] = totalLength;
  assert(totalLength <= lengths->dims()[0] && "sum(Lengths) must be equal to len(indices)");

  std::fill(outH.getIterator(offset), outH.getIterator(offset + maxRead), 0);

  // Obtain the boundaries of the segments/elements proceesed by this minion.
  auto firstSegment = offset / segmentStride;
  auto lastSegment = (offset + maxRead - 1) / segmentStride;
  // Some partition offsets might fall halfway of a segment.
  auto firstSegmentBegin = offset % segmentStride; // Index of the first element to process
  auto lastSegmentEnd =
    std::min(numColumns - 1, (offset + maxRead - 1) % segmentStride); // Index of the last element to process

  auto idx = idxH.getIterator(segmentPtr[firstSegment]);
  auto out = outH.getIterator(offset);
  auto len = lengthH.getIterator(firstSegment);
  dim_array_t inCoords = {0};

  for (size_t s = firstSegment; s <= lastSegment; s++, out.step(0), ++len) {
    size_t firstElem = (s == firstSegment) ? firstSegmentBegin : 0;
    size_t lastElem = (s == lastSegment) ? lastSegmentEnd + 1 : numColumns;
    for (int32_t j = 0; j < *len; j++, ++idx) {
      auto outIt = out;
      inCoords[0] = *idx;
      auto dataIt = dataH.getIterator(inCoords);
      dataIt += static_cast<std::ptrdiff_t>(firstElem);
      for (size_t k = firstElem; k < lastElem; k++) {
        *outIt += (*dataIt);
        ++dataIt;
        ++outIt;
      }
    }
  }
  outT->evict(DO_EVICTS, offset, maxRead);
}

} // inlining
} // namespace

#endif
