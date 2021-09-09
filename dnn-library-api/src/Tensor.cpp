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

#include "dnnLibraryApi/LibTensor.h"

namespace dnn_lib {

Type Type::newShape(const Type& T, size_t numSizes, const dim_array_t& dims, const dim_array_t& pitches) {
  if (T.isQuantizedType()) {
    return Type(T.elementType_, numSizes, dims, pitches, T.scale_, T.offset_);
  } else {
    return Type(T.elementType_, numSizes, dims, pitches);
  }
}

Type Type::newShape(const Type& kindType, const Type shapeType) {
  //@TODO  T.getElementType() == shapeType->getelementSize() Size should be the same
  if (kindType.isQuantizedType()) {
    return Type(kindType.elementType_, shapeType.sizes_, shapeType.strides_, kindType.scale_, kindType.offset_);
  } else {
    return Type(kindType.elementType_, shapeType.sizes_, shapeType.strides_);
  }

  // TODO: the numSizes_is set wrong => because of dimension and strides extension. Either set properly (e.g. separate
  // extended
  // and non extended arrays... or maybe just delete this newShape, in case it is not needed)
}

const bool Type::hasSameShape(const Type other) const {
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
  return isQuantizedElemKind(elementType_);
}

bool Type::isIndexType() const {
  return isIndexElemKind(elementType_);
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
  return type_.numSizes_;
}

const dim_array_t& LibTensor::dims() const {
  return type_.sizes_;
}

const dim_array_t& LibTensor::strides() const {
  return type_.strides_;
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

float LibTensor::getScale() const {
  return type_.getScale();
}

int32_t LibTensor::getOffset() const {
  return type_.getOffset();
}

size_t LibTensor::getElementSize() const {
  return type_.getElementSize();
}
// end LibTensor class

} // end namespace dnn_lib