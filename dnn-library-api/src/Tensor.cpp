/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#include "LibTensor.h"

namespace dnn_lib {

Type::Type(const dnn_lib::ElemKind elk, const size_t numSizes, const dim_array_t& dims, const dim_array_t& strides,
           const float scale, const int32_t offset, const bool hasSingleValue, const float singleValue,
           const bool isCounter, const int64_t counterOffset, const int64_t counterStride)
  : sizes_(dims)
  , strides_(strides)
  , elementType_(elk)
  , numSizes_(numSizes)
  , scale_(scale)
  , offset_(offset)
  , hasSingleValue_(hasSingleValue)
  , singleValue_(singleValue)
  , isCounter_(isCounter)
  , counterOffset_(counterOffset)
  , counterStride_(counterStride) {
}

Type::Type(const dnn_lib::ElemKind elk, const size_t numSizes, const dim_array_t& dims, const dim_array_t& strides,
           const float scale, const int32_t offset)
  : sizes_(dims)
  , strides_(strides)
  , elementType_(elk)
  , numSizes_(numSizes)
  , scale_(scale)
  , offset_(offset) {
}

Type::Type(const dnn_lib::ElemKind elk, const size_t numSizes, const dim_array_t& dims, const dim_array_t& strides)
  : sizes_(dims)
  , strides_(strides)
  , elementType_(elk)
  , numSizes_(numSizes) {
  assert(not isQuantizedElemKind());
}

bool Type::hasSameShape(const Type other) const {
  if (numSizes_ != other.getNumDims())
    return false;
  const dim_array_t& other_sizes = other.getSizes();
  const dim_array_t& other_strides = other.getStrides();
  for (size_t idx = 0; idx < numSizes_; idx++) {
    if (sizes_[idx] != other_sizes[idx]) {
      return false;
    }
    if (strides_[idx] != other_strides[idx]) {
      return false;
    }
  }
  return true;
}

float Type::getScale() const {
  //@TODO assert(isQuantizedType() && "Can't get the scale of non-quantized type");
  return scale_;
}

int32_t Type::getOffset() const {
  //@TODO assert(isQuantizedType() && "Can't get the offset of a non-quantized type");
  return offset_;
}

const dim_array_t& Type::getSizes() const {
  return sizes_;
}

const dim_array_t& Type::getStrides() const {
  return strides_;
}

dim_t Type::getNumDims() const {
  return numSizes_;
}

ElemKind Type::getElementType() const {
  return elementType_;
}

const dim_t Type::size() const {
  dim_t acum = 1;
  for (auto i : sizes_)
    acum *= i;
  return acum;
}

bool Type::isQuantizedType() const {
  return isQuantizedElemKind();
}

bool Type::isIndexType() const {
  return isIndexElemKind();
}

unsigned Type::getElementSize() const {
  return getElementSize(elementType_);
}

size_t Type::getSizeInBytes() const {
  return sizes_[0] * strides_[0] * getElementSize();
}

size_t Type::actualSize() const {
  return (sizes_[0] * strides_[0]);
}

bool Type::isQuantizedElemKind() const {
  if ((elementType_ == dnn_lib::ElemKind::Int8QTy) or (elementType_ == dnn_lib::ElemKind::UInt8QTy) or
      (elementType_ == dnn_lib::ElemKind::Int16QTy) or (elementType_ == dnn_lib::ElemKind::Int32QTy) or
      (elementType_ == dnn_lib::ElemKind::UInt8FusedQTy) or (elementType_ == dnn_lib::ElemKind::UInt8FusedFP16QTy) or
      (elementType_ == dnn_lib::ElemKind::UInt4FusedFP16QTy)) {
    return true;
  } else {
    return false;
  }
}

bool Type::isIndexElemKind() const {
  return (elementType_ == dnn_lib::ElemKind::Int32ITy) or (elementType_ == dnn_lib::ElemKind::Int64ITy);
}

bool Type::hasSingleValue() const {
  return hasSingleValue_;
}

float Type::getSingleValue() const {
  assert(hasSingleValue() and "Single value can only be obtained when it is set");
  return singleValue_;
}

bool Type::isCounter() const {
  return isCounter_;
}

int64_t Type::getCounterOffset() const {
  return counterOffset_;
}

int64_t Type::getCounterStride() const {
  return counterStride_;
}

// end Type class

///////

char* LibTensor::getAddress() const {
  return ptrData_;
}

const Type& LibTensor::getType() const {
  return type_;
}

const ElemKind LibTensor::getElementType() const {
  return type_.getElementType();
}

const dim_t LibTensor::ndims() const {
  return type_.getNumDims();
}

const dim_array_t& LibTensor::dims() const {
  return type_.getSizes();
}

const dim_array_t& LibTensor::strides() const {
  return type_.getStrides();
}

const dim_array_t LibTensor::stridesNoPadding() const {
  dim_array_t v;
  v[ndims() - 1] = 1;
  for (int64_t i = ndims() - 2; i >= 0; i--) {
    v[i] = v[i + 1] * dims()[i + 1];
  }
  return v;
}

dim_t LibTensor::size() const {
  return type_.size();
}

dim_t LibTensor::actualSize() const {
  return type_.actualSize();
}

uint64_t LibTensor::getSizeInBytes() const {
  return type_.getSizeInBytes();
}

LibTensor::LibTensor(const Type& type, void* const rawdata, const bool untouch)
  : ptrData_(reinterpret_cast<char*>(rawdata))
  , type_(type)
  , untouch_(untouch) {
}

LibTensor::LibTensor(const Type&& type, void* const rawdata, const bool untouch)
  : ptrData_(reinterpret_cast<char*>(rawdata))
  , type_(std::move(type))
  , untouch_(untouch) {
}

LibTensor::LibTensor(const Tensor& tensor)
  : ptrData_(reinterpret_cast<char*>(tensor.alignOffset))
  , type_(tensor.elementType, tensor.numDims, tensor.sizes, tensor.strides, tensor.scale, tensor.offset,
          tensor.hasSingleValue, tensor.singleValue, tensor.isCounter, tensor.counterOffset, tensor.counterStride)
  , untouch_(tensor.untouchablePadding) {
}

float LibTensor::getScale() const {
  return type_.getScale();
}

int32_t LibTensor::getOffset() const {
  return type_.getOffset();
}

size_t LibTensor::getElementSize() const {
  return type_.getElementSize();
}

bool LibTensor::getUntouchable() const {
  return untouch_;
}

bool LibTensor::hasSingleValue() const {
  return type_.hasSingleValue();
}

float LibTensor::getSingleValue() const {
  return type_.getSingleValue();
}

bool LibTensor::isCounter() const {
  return type_.isCounter();
}

int64_t LibTensor::getCounterOffset() const {
  return type_.getCounterOffset();
}

int64_t LibTensor::getCounterStride() const {
  return type_.getCounterStride();
}

// end LibTensor class

} // end namespace dnn_lib
