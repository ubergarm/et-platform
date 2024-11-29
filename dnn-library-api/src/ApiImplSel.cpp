/*-------------------------------------------------------------------------
 * Copyright (C) 2021, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include "LibApiImplSel.h"
#include <algorithm>

namespace dnn_lib {

#define CACHE_LINE_BYTES 64

size_t implSel::ResizeBilinear(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  // Check for (1,1,2,2) upscaling and tensor type
  if ((inTensors[0]->ndims() != 4) or (outTensors[0]->getElementType() != Float16Ty)) {
    return 0; // default impl
  }
  // Check that in and out tensors are cache aligned
  if ((((uintptr_t)(inTensors[0]->getAddress()) & (uintptr_t)(~(CACHE_LINE_BYTES - 1))) != 0) or
      (((uintptr_t)(outTensors[0]->getAddress()) & (uintptr_t)(~(CACHE_LINE_BYTES - 1))) != 0)) {
    return 0;
  }
  // Check for (1*b,2*h,2*w,1*c) upscaling
  auto dimsIn = inTensors[0]->dims();
  auto dimsOut = outTensors[0]->dims();
  if (dimsOut[0] == dimsIn[0] and dimsOut[1] == 2 * dimsIn[1] and dimsOut[2] == 2 * dimsIn[2] and
      dimsOut[3] == dimsIn[3]) {
    return 1;
  }
  return 0;
}

size_t implSel::ResizeNearest(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  // check conditions for applying fast resize
  // Initial address has to be aligned to 16 bits for source and 32 for dest
  if ((((uintptr_t)inTensors[0]->getAddress() & 0xf) != 0) or ((uintptr_t)outTensors[0]->getAddress() & 0x1f) != 0) {
    return 0;
  }
  auto ndims = inTensors[0]->ndims();
  auto dimsIn = inTensors[0]->dims();
  auto dimsOut = outTensors[0]->dims();
  auto ratio = static_cast<float>(dimsOut[ndims - 1]) / static_cast<float>(dimsIn[ndims - 1]);
  // Last dimension must be upscaled by 2
  if (std::abs(ratio - 2) > 1e-15) {
    return 0;
  }
  // Every other dimension must be upscaled by an integer
  for (size_t dim = 0; dim < ndims - 1; dim++) {
    float dimRatio = static_cast<float>(dimsOut[dim]) / static_cast<float>(dimsIn[dim]);
    if (std::abs(int(dimRatio) - dimRatio) > 1e-15) {
      return 0;
    }
  }
  auto typeSize = inTensors[0]->getElementSize();
  auto strideIn = inTensors[0]->strides().data();
  auto stridesOut = outTensors[0]->strides().data();
  // All rows in last dimension must start in an address divisible by 16 for src and 32 for dest
  if (((strideIn[ndims - 2] * typeSize & 0xf) != 0) or ((stridesOut[ndims - 2] * typeSize & 0x1f) != 0)) {
    return 0;
  }
  return 1;
}

size_t implSel::Copy(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  // Tensorized only works with same shape in-out and CL aligment
  if (inTensors[0]->getType().hasSameShape(outTensors[0]->getType()) and
      (((uintptr_t)inTensors[0]->getAddress() & 0x3F) == 0) and
      ((inTensors[0]->getType().getSizeInBytes() & 0x3F) == 0) and
      (((uintptr_t)outTensors[0]->getAddress() & 0x3F) == 0) and
      ((outTensors[0]->getType().getSizeInBytes() & 0x3F) == 0) and
      ((outTensors[0]->getUntouchable() == false) || (outTensors[0]->size() == outTensors[0]->actualSize()))) {
    return 1;
  } else {
    return 0;
  }
}

size_t implSel::ElementBool(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  // SW-6832: Support source operands with different padding in vectorized ElementBool
  ElemKind src1ElK = inTensors[0]->getElementType();
  LibTensor* in1T = inTensors[0];
  LibTensor* in2T = inTensors[1];
  if (((src1ElK == ElemKind::FloatTy) or (src1ElK == ElemKind::Float16Ty) or (src1ElK == ElemKind::Int8QTy)) and
      (in1T->strides()[0] == in2T->strides()[0])) {
    return 1;
  } else {
    return 0;
  }
}

size_t implSel::ElementCmpEQ(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  return ElementBool(outTensors, inTensors);
}

size_t implSel::ElementCmpNEQ(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  return ElementBool(outTensors, inTensors);
}

size_t implSel::ElementCmpLTE(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  return ElementBool(outTensors, inTensors);
}

size_t implSel::ElementCmpLT(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  return ElementBool(outTensors, inTensors);
}

size_t implSel::MaxSplat(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  // SW-6834: Exploit the case when only one MaxSplat operand is aligned to 32 bytes
  LibTensor* outT = outTensors[0];
  LibTensor* inT = inTensors[0];
  const static size_t batchDim = inT->ndims() - 2;
  if ((inT->ndims() >= 2) and (((uintptr_t)inT->getAddress() % 32) == 0) and
      (((uintptr_t)outT->getAddress() % 32) == 0) and
      (((outT->strides()[batchDim] % 32) == 0) or ((32 % outT->strides()[batchDim]) == 0)) and
      (((inT->strides()[batchDim] % 32) == 0) or ((32 % inT->strides()[batchDim]) == 0))) {
    return 1;
  } else {
    return 0;
  }
}

size_t implSel::RowwiseQuantizedFullyConnected(std::vector<LibTensor*>& outTensors,
                                               std::vector<LibTensor*>& inTensors) {
  // SW-6838: Exploit all the operand alignment combinations for RowwiseQuantizedFullyConnected
  return 0; // Workaround for SW-12216 !
  LibTensor* outT = outTensors[0];
  LibTensor* in1T = inTensors[0];
  LibTensor* in2T = inTensors[1];

  const static size_t batchDim = in1T->ndims() - 2;
  if ((((uintptr_t)outT->getAddress() % 32) == 0) and (((uintptr_t)in1T->getAddress() % 32) == 0) and
      (((uintptr_t)in2T->getAddress() % 32) == 0) and ((outT->strides()[batchDim] % 32) == 0) and
      ((in1T->strides()[batchDim] % 32) == 0) and ((in2T->strides()[batchDim] % 32) == 0)) {
    return 1;
  } else {
    return 0;
  }
}

size_t implSel::RowwiseQuantizedSparseLengthsWeightedSum(std::vector<LibTensor*>& outTensors,
                                                         std::vector<LibTensor*>& inTensors) {
  // SW-3119: RowwiseQSL(W)S Operator tests failing
  LibTensor* dataT = inTensors[0];
  if (dataT->dims()[dataT->ndims() - 1] < 4) {
    return 0;
  } else {
    return 1;
  }
}

size_t implSel::Transpose(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  LibTensor* outT = outTensors[0];
  LibTensor* inT = inTensors[0];
  ElemKind elK = inTensors[0]->getElementType();
  const size_t batchDim = inT->ndims() - 2;
  if ((inT->ndims() >= 2) and (((uintptr_t)outT->getAddress() % 32) == 0) and
      (((uintptr_t)inT->getAddress() % 32) == 0) and
      (((outT->strides()[batchDim] * outT->getElementSize()) % 32) == 0) and
      (((inT->strides()[batchDim] * inT->getElementSize()) % 32) == 0) and
      ((elK == ElemKind::FloatTy) or (elK == ElemKind::Float16Ty) or (elK == ElemKind::Int8QTy))) {
    return 1;
  } else {
    return 0;
  }
}

size_t implSel::SoftMax(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  // SW-6840: Vectorized SoftMax with unconstrained destination alignment
  LibTensor* outT = outTensors[0];
  unsigned cll = CACHE_LINE_BYTES / outT->getElementSize();
  const size_t numDims = outT->ndims();
  if ((((uintptr_t)outT->getAddress() % CACHE_LINE_BYTES) == 0) and (numDims >= 2) and
      ((outT->strides()[numDims - 2] % cll) == 0)) {
    return 1;
  } else {
    return 0;
  }
}

size_t implSel::LocalResponseNormalization(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  // FIXME: [SW-10889] Vectorized impl may end up in  Write-coherency-errors/ numerical erros on some dimensions.
  return 0;
}

size_t implSel::SparseLengthsSum([[maybe_unused]] std::vector<LibTensor*>& outTensors,
                                 [[maybe_unused]] std::vector<LibTensor*>& inTensors) {
  return 0;
}

size_t implSel::SparseLengthsWeightedSum(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  return 0;
}

size_t implSel::InsertTensor(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  auto dstTypeSize = outTensors[0]->getElementSize();
  auto dstCll = CACHE_LINE_BYTES / dstTypeSize;
  auto& dstPitch = outTensors[0]->strides();
  auto dstDimNum = outTensors[0]->ndims();

  if (((uintptr_t)outTensors[0]->getAddress() % CACHE_LINE_BYTES) != 0) {
    return 0;
  }

  if ((dstDimNum < 2) or ((dstPitch[dstDimNum - 2] % dstCll) != 0)) {
    return 0;
  }

  return 1;
}

size_t implSel::AvgPool(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors) {
  return (inTensors[0]->ndims() == 5) ? 0 : 1;
}

size_t implSel::EmbeddingBag([[maybe_unused]] std::vector<LibTensor*>& outTensors,
                             [[maybe_unused]] std::vector<LibTensor*>& inTensors) {

  constexpr float epsilon_i8 = 2.0f / 256.0f;
  LibTensor* dataT = inTensors[0];
  LibTensor* weightT = inTensors[1];
  LibTensor* offsetT = inTensors[3];

  bool isCounter = offsetT->isCounter() && offsetT->getCounterOffset() == 0 && offsetT->getCounterStride() == 1;
  bool isDataQuantized = dataT->getElementType() == ElemKind::Int8QTy;
  bool isWeightFloatOne = weightT->hasSingleValue() && weightT->getSingleValue() == 1.0f;
  bool isWeightQuantizedOne = weightT->hasSingleValue() && weightT->getElementType() == ElemKind::Int8QTy &&
                              (std::abs(weightT->getSingleValue() - 1.0f) < epsilon_i8);
  bool isWeightOne = isWeightFloatOne || isWeightQuantizedOne;

  if (isCounter && isWeightOne && isDataQuantized) {
    return 2;
  } else {
    return 1;
  }
}

} // end namespace dnn_lib
