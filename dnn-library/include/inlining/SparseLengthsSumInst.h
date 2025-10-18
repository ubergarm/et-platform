/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
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
 * tensor. Summation is done element-wise across slices of the input tensor and does not
 * change the shape of the individual blocks
 */

/// \brief Accumulates element-wise the input vector onto the output vector.
///
/// This code is not vectorized.
///
/// \tparam elKind Element kind of input and output tensors
/// \tparam onlyCopy If enabled only copies input values, useful if output values are not initialized or 0.
/// \tparam T elKind datatype
/// \param[in] in Input tensor iterator
/// \param[in] out Output tensor iterator
/// \param[in] n Number of (elKind) elements to accumulate
template <ElemKind elKind, bool onlyCopy = false, typename T = typename elemKind2elemTy<elKind>::type>
INLINE_ATTR typename std::enable_if_t<(isQuantizedElemKind(elKind)), void>
vectorAccumulate(HandleIterator<T>& in, HandleIterator<T>& out, const size_t n, const int32_t offsetIn,
                 const int32_t offsetOut, const float scaleIn, const float scaleOut) {
  using elkType = typename elemKind2elemTy<elKind>::type;

  for (size_t i = 0; i < n; i++) {
    if constexpr (onlyCopy) {
      // Do not accumulate, just copy from (in) to (out)
      float dataValue = dequantize<elkType>(*in, scaleIn, offsetIn);
      (*out) = quantize<elkType>(dataValue, scaleOut, offsetOut);
    } else {
      // Acumulate Quantized
      float dataValue = dequantize<elkType>(*in, scaleIn, offsetIn);
      float accumValue = dequantize<elkType>(*out, scaleOut, offsetOut);
      accumValue += dataValue;
      (*out) = quantize<elkType>(accumValue, scaleOut, offsetOut);
    }
    ++in;
    ++out;
  }
}

template <ElemKind elKind, bool onlyCopy = false, typename T = typename elemKind2elemTy<elKind>::type>
INLINE_ATTR typename std::enable_if_t<(!isQuantizedElemKind(elKind)), void>
vectorAccumulate(HandleIterator<T>& in, HandleIterator<T>& out, const size_t n, [[maybe_unused]] const int32_t offsetIn,
                 [[maybe_unused]] const int32_t offsetOut, [[maybe_unused]] const float scaleIn,
                 [[maybe_unused]] const float scaleOut) {
  for (size_t i = 0; i < n; i++) {
    if constexpr (onlyCopy) {
      // Do not accumulate, just copy from (in) to (out)
      (*out) = *in;
    } else {
      if constexpr (elKind == Float16Ty) {
        // Acumulate FP16
        uint16_t accumValueFP16 = 0;
        float dataValue, accumValue;
        convertFp16ToFp32(static_cast<uint16_t>(*in), dataValue);
        convertFp16ToFp32(static_cast<uint16_t>(*out), accumValue);
        accumValue += dataValue;
        convertFp32ToFp16(accumValue, accumValueFP16);
        (*out) = accumValueFP16;
      } else {
        // Acumulate any other datatype
        (*out) += *in;
      }
    }
    ++in;
    ++out;
  }
}

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
INLINE_ATTR void fwdLibSparseLengthsSumInstThreaded(LibTensor* outT, LibTensor* data, LibTensor* indices,
                                                    LibTensor* lengths, uint64_t flags, const uint32_t minionOffset = 0,
                                                    const uint32_t assignedMinions = 0) {

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  assert((data->getElementType() == outT->getElementType()) && "Input and Ouput tensor datatypes must the same");
  assert((indices->getElementType() == Int64ITy) ||
         ((indices->getElementType() == Int32ITy) && "indices datatype must be integer 32/64-bit"));
  assert((lengths->getElementType() == Int32ITy) && "Lenghts datatype must be 32-bit integer");

  // Obtain data partition using the output tensor
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
  const dim_t nSegments = outT->dims()[0];
  const dim_t nColumns = outT->dims()[1];

  // Partitioning Logic suitable for multi-op: see SW-11826
  //
  // Does this minion has useful work to do or just got assigned a padding region?
  auto firstElemOffset = offset % segmentStride;                // Index of the first element to process,
  auto lastElemOffset = (offset + maxRead - 1) % segmentStride; // and the last.
  // Segments to process in this minion.
  auto firstSegment = offset / segmentStride;
  auto lastSegment = (offset + maxRead - 1) / segmentStride;
  // Offsets correspond to padding?
  bool startsAtPadding = firstElemOffset > nColumns;
  bool endsAtPadding = lastElemOffset >= nColumns;

  if (startsAtPadding && endsAtPadding) {
    if (firstSegment == lastSegment) {
      return; // minion has no USEFUL work to do
      // This is not optimal, we observed some models where 75% of the minions assigned do not have anything to do.
      // TODO: discuss a new partitionCL that avoids padding completely.
    }
  }

  // Post: segmentPtr[i] contains a pointer to the first index corresponding to segment i.
  size_t segmentPtr[nSegments + 1];
  size_t totalLength = 0;
  for (size_t i = 0; i < nSegments; i++) {
    segmentPtr[i] = totalLength;
    totalLength += lengthH.raw(i);
  }
  segmentPtr[nSegments] = totalLength;
  assert(totalLength <= lengths->dims()[0] && "sum(Lengths) must be equal to len(indices)");

  // Setup starting iterators to data structures
  auto idx = idxH.getIterator(segmentPtr[firstSegment]);
  auto out = outH.getIterator(offset);
  auto len = lengthH.getIterator(firstSegment);
  dim_array_t dataCoords = {0};

  // move output iterator and offsets to avoid padding
  if (startsAtPadding) {
    out.step(0);
    firstElemOffset = 0;
  }
  if (endsAtPadding) {
    lastElemOffset = nColumns - 1;
  }

  // SparseLengthSum compute loop
  for (size_t s = firstSegment; s <= lastSegment; s++, out.step(0), ++len) {
    size_t startElem = (s == firstSegment) ? firstElemOffset : 0;
    size_t lastElem = lastElemOffset + 1;

    size_t numElems = lastElem - startElem;
    if (unlikely(*len == 0)) {
      auto outIt = out;
      // If segment.len == 0, fill output segment with 0
      for (size_t k = startElem; k < lastElem; k++) {
        (*outIt) = 0;
        ++outIt;
      }
    } else {
      // j==0; copy first slice of data to the output segment
      auto outIt = out;
      dataCoords[0] = *idx;
      auto dataIt = dataH.getIterator(dataCoords);
      dataIt += static_cast<std::ptrdiff_t>(startElem);
      vectorAccumulate<elKind, true>(dataIt, outIt, numElems, data->getOffset(), outT->getOffset(), data->getScale(),
                                     outT->getScale());
      ++idx;

      // Accumulate the remaining slices
      for (size_t j = 1; j < *len; j++, ++idx) {
        auto outIt2 = out;
        dataCoords[0] = *idx;
        auto dataIt2 = dataH.getIterator(dataCoords);
        dataIt += static_cast<std::ptrdiff_t>(startElem);
        vectorAccumulate<elKind>(dataIt2, outIt2, numElems, data->getOffset(), outT->getOffset(), data->getScale(),
                                 outT->getScale());
      }
    }
  }
  outT->evict(DO_EVICTS, offset, maxRead);
}

} // inlining
} // namespace

#endif
