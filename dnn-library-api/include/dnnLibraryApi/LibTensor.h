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

#ifndef LIB_TENSOR_H
#define LIB_TENSOR_H

#include "LibTypes.h"
#include "LibUtils.h"
#include <cassert>
#include <cstring>
#include <numeric>
#include <tuple>
#ifdef __riscv
#include <device-common/cacheops.h>
#endif

namespace dnn_lib {

struct Type final {

  /*@brief contains the dimensions (sizes) of the tensor.
   */
  const dim_array_t sizes_;

  /*@brief contains the strides for each dimension (in elements) same order
   * as in sizes_.
   */
  const dim_array_t strides_;

  /*@brief Specifies the element type of the tensor.
   */
  const dnn_lib::ElemKind elementType_{dnn_lib::ElemKind::Int64ITy};

  /*@brief contains the number of dimensions used by the tensor.
   */
  const dim_t numSizes_;

  /*@brief On quantized tensors, this represents the scale of the values.
   */
  const float scale_ {};

  /*@brief On quantized tensors, this represents the offset of the values.
   */
  const int32_t offset_ {};

  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  template<size_t numSizes>
  constexpr Type(const dnn_lib::ElemKind elk, const std::array<dim_t, numSizes> &dims, const float scale, const int32_t offset) :
    sizes_(make_dims(dims)),
    elementType_(elk), numSizes_(numSizes),
    scale_(scale), offset_(offset)
  {
    assert(isQuantizedElemKind(elk));
  }

  /*@brief Initialize a new non-quantized type.
   */
  template<size_t numSizes>
  constexpr Type(const dnn_lib::ElemKind elk, const std::array<dim_t, numSizes> &dims) :
    sizes_(make_dims(dims)),
    elementType_(elk), numSizes_(numSizes)
  {
    assert(not isQuantizedElemKind(elk));
  }

  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  template<size_t numSizes>
  constexpr Type(const dnn_lib::ElemKind elk,  const std::array<dim_t, numSizes> &dims, const std::array<dim_t, numSizes> &strides,
                 const float scale, const int32_t offset) :
    sizes_(make_dims(dims)),
    strides_(make_strides(strides)),
    elementType_(elk),
    numSizes_(numSizes),
    scale_(scale), offset_(offset)
  {
    assert(isQuantizedElemKind(elk));
  }

  /*@brief Initialize a new non-quantized type.
   */
  template<size_t numSizes>
  constexpr Type(const dnn_lib::ElemKind elk,  const std::array<dim_t, numSizes> &dims, const std::array<dim_t, numSizes> &strides) :
    sizes_(make_dims(dims)),
    strides_(make_strides(strides)),
    elementType_(elk),
    numSizes_(numSizes)
  {
    assert(not isQuantizedElemKind(elk));
  }

  /*@brief non templated version of the previous constructors (receiving dimensions/strides with max_tensor_dimensions,
    and an extra parameter to set the actual number of dimensions
   */
  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  constexpr Type(const dnn_lib::ElemKind elk, const size_t numSizes, const dim_array_t &dims, const dim_array_t &strides,
                 const float scale, const int32_t offset) :
    sizes_(dims),
    strides_(strides),
    elementType_(elk),
    numSizes_(numSizes),
    scale_(scale), offset_(offset)
  {
    assert(isQuantizedElemKind(elk));
  }

  /*@brief Initialize a new non-quantized type.
   */
  constexpr Type(const dnn_lib::ElemKind elk, const size_t numSizes, const dim_array_t &dims, const dim_array_t &strides) :
    sizes_(dims),
    strides_(strides),
    elementType_(elk),
    numSizes_(numSizes)
  {
    assert(not isQuantizedElemKind(elk));
  }

  /*@brief Reshape existing type this takes care of quantized types.
   */
  template<size_t numSizes>
  static Type newShape(const Type &T, const std::array<dim_t, numSizes> &dims) {
    if (T.isQuantizedType()) {
      return Type(T.elementType_, dims, T.scale_, T.offset_);
    } else {
      return Type(T.elementType_, dims);
    }
  }

  /*@brief Reshape existing type and change alignments.
    */
  template<size_t numSizes>
  static Type newShape(const Type &T, const std::array<dim_t, numSizes> &dims, const std::array<dim_t, numSizes> &pitches){
    if (T.isQuantizedType()) {
      return Type(T.elementType_, dims, pitches, T.scale_, T.offset_);
    } else {
      return Type(T.elementType_, dims, pitches);
    }
  }

  static Type newShape(const Type &T, size_t numSizes, const dim_array_t &dims, const dim_array_t &pitches){
    if (T.isQuantizedType()) {
      return Type(T.elementType_, numSizes, dims, pitches, T.scale_, T.offset_);
    } else {
      return Type(T.elementType_, numSizes, dims, pitches);
    }
  }

  /*@brief Reshape existing type by taking shapes and strides of \p shapeType.
   */
  static Type newShape(const Type &kindType, const Type shapeType) {
    //@TODO  T.getElementType() == shapeType->getelementSize() Size should be the same
    if (kindType.isQuantizedType()) {
      return Type(kindType.elementType_, shapeType.sizes_, shapeType.strides_, kindType.scale_, kindType.offset_);
    } else {
      return Type(kindType.elementType_, shapeType.sizes_, shapeType.strides_);
    }

    //TODO: the numSizes_is set wrong => because of dimension and strides extension. Either set properly (e.g. separate extended
    // and non extended arrays... or maybe just delete this newShape, in case it is not needed)
  }

  /* brief returns true if \p other has same shape.
   */
  const bool hasSameShape(const Type other) const {
    if (numSizes_ != other.getNumDims()) return false;
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

  /*@brief returns the scale of a quantized type.
   */
  float getScale() const {
    //@TODO assert(isQuantizedType() && "Can't get the scale of non-quantized type");
    return scale_;
  }

  /*@brief returns the offset of quantized type.
   */
  int32_t getOffset() const {
    //@TODO assert(isQuantizedType() && "Can't get the offset of a non-quantized type");
    return offset_;
  }

  /*@brief returns the Tensor sizes_.
   */
  const dim_array_t& getSizes() const {
    return sizes_;
  }

  /*@brief returns the Tensor strides_.
   */
  const dim_array_t& getStrides() const {
    return strides_;
  }

  /*@brief returns the Tensor dimension.
   */
  dim_t getNumDims() const { return numSizes_;}


  /*@brief returns the elemet type
   */
  ElemKind getElementType() const { return elementType_; }

  /*@brief return the number of elements in the tensor.
   */
  const dim_t size() const {
    dim_t acum = 1;
    for(auto i: sizes_) acum*=i;
    return acum;
  }

  /*@brief returns true if the templated parameter \p Elemkind matches this type.
   */
  template<class ElemTy> bool isType() const {
    return isType<ElemTy>(elementType_);
  }

  /*@brief returns true if the templated parameter \p ElemKind matches the type
   *that's specified by the parameter \p Ty.
   */
  template<class ElemTy> static bool isType(dnn_lib::ElemKind elk) {
    switch (elk) {
    case dnn_lib::ElemKind::FloatTy:
      return std::is_same<ElemTy, float>::value;
    case dnn_lib::ElemKind::Float16Ty:
      return std::is_same<ElemTy, uint16_t>::value;
    case dnn_lib::ElemKind::BFloat16Ty:
      return std::is_same<ElemTy, uint16_t>::value;
    case dnn_lib::ElemKind::Int8QTy:
      return std::is_same<ElemTy, int8_t>::value;
    case dnn_lib::ElemKind::UInt8QTy:
      return std::is_same<ElemTy, uint8_t>::value;
    case dnn_lib::ElemKind::Int16QTy:
      return std::is_same<ElemTy, int16_t>::value;
    case dnn_lib::ElemKind::Int32QTy:
      return std::is_same<ElemTy, int32_t>::value;
    case dnn_lib::ElemKind::Int32ITy:
      return std::is_same<ElemTy, int32_t>::value;
    case dnn_lib::ElemKind::Int64ITy:
      return std::is_same<ElemTy, int64_t>::value;
    case dnn_lib::ElemKind::UInt8FusedQTy:
      return std::is_same<ElemTy, uint8_t>::value;
    case dnn_lib::ElemKind::UInt8FusedFP16QTy:
      return std::is_same<ElemTy, uint8_t>::value;
    case dnn_lib::ElemKind::UInt4FusedFP16QTy:
      return std::is_same<ElemTy, uint8_t>::value;
    case dnn_lib::ElemKind::UInt4FusedQTy:
      return std::is_same<ElemTy, uint8_t>::value;
    case dnn_lib::ElemKind::BoolTy:
      return std::is_same<ElemTy, bool>::value;
    }
    assert(true && "Invalid type");
    __builtin_unreachable();
  }

  /*@brief true if the type of this Tensor is one of the quantized
   *types.
   */
  bool isQuantizedType() const {
    return isQuantizedElemKind(elementType_);
  }

  /*@brief true if the type of this Tensor is one of the and index type
   */
  bool isIndexType() const {
    return isIndexElemKind(elementType_);
  }

  /*@brief returns the size of the type element.
   */
  unsigned getElementSize() const {
    return getElementSize(elementType_);
  }

  /*@brief returns the size in bytes for this Tensor.
   */
  size_t getSizeInBytes() const {
    return sizes_[0] * strides_[0] * getElementSize();
  }

  /*@brief the actual number of elements in the tensor taking striding into
   * account. Since size() does not take striding into account, size() is
   * always <= actualSize()
   */
  size_t actualSize() const {
    return (sizes_[0] * strides_[0]);
  }

  /// \return the size of the element \p Ty.
  static constexpr size_t getElementSize(dnn_lib::ElemKind Ty) {
    switch (Ty) {
    case dnn_lib::ElemKind::FloatTy:
      return sizeof(float);
    case dnn_lib::ElemKind::Float16Ty:
      return sizeof(uint16_t);
    case dnn_lib::ElemKind::BFloat16Ty:
      return sizeof(uint16_t);
    case dnn_lib::ElemKind::Int8QTy:
      return sizeof(int8_t);
    case dnn_lib::ElemKind::UInt8QTy:
      return sizeof(uint8_t);
    case dnn_lib::ElemKind::Int16QTy:
      return sizeof(int16_t);
    case dnn_lib::ElemKind::Int32QTy:
      return sizeof(int32_t);
    case dnn_lib::ElemKind::Int32ITy:
      return sizeof(int32_t);
    case dnn_lib::ElemKind::Int64ITy:
      return sizeof(int64_t);
    case dnn_lib::ElemKind::UInt8FusedQTy:
      return sizeof(uint8_t);
    case dnn_lib::ElemKind::UInt8FusedFP16QTy:
      return sizeof(uint8_t);
    case dnn_lib::ElemKind::UInt4FusedFP16QTy:
      return sizeof(uint8_t);
    case dnn_lib::ElemKind::UInt4FusedQTy:
      return sizeof(uint8_t);
    case dnn_lib::ElemKind::BoolTy:
      return sizeof(bool);
    }
    assert(true && "Invalid type");
    __builtin_unreachable();
  }

}; //class Type

class LibTensor final {
private:
  char* const ptrData_;
  const Type type_;
  const bool untouch_;

public:
  /* @brief returns the start address of the tensor.
   */
  char* getAddress() const {
    return ptrData_;
  }

  /* @brief returns the type of the tensor.
   */
  const Type& getType() const {
    return type_;
  }

  /*@brief returns the element type of the tensor.
   */
  const ElemKind getElementType() const {
    return type_.getElementType();
  }

  /*@brief Get number of dimensions the tensor has
   */
  const dim_t ndims() const {
    return type_.numSizes_;
  }

  /*@brief returns the dimensions (padded with 1 until max_tensor_dimensions)
   */
  const dim_array_t& dims() const {
    return type_.sizes_;
  }

  /*@brief returns the strides (padded with 0 until max_tensor_dimensions)
   */
  const dim_array_t& strides() const {
    return type_.strides_;
  }

  /*@brief returns strides as if there were no padding
   */
  const dim_array_t stridesNoPadding() const {
    dim_array_t v;
    v[ndims()-1] = 1;
    for (int64_t i = ndims()-2; i >=0; i--){
      v[i] = v[i+1] * dims()[i+1];
    }
    return v;
  }

  /*@brief returns the number of real menaingful elements in the tensor. Does
   *not take strides into account.
   */
  dim_t size() const {
    return type_.size();
  }

  /*@brief returns the actaul number of elements in the tensor taking stridding
   *into account. Since size() does not take striding into account, size() is
   *always <= actualSize(),
   */
  dim_t actualSize() const {
    return type_.actualSize();
  }

  /*@brief returns the number of bytes required to store the tensor based on its
   *Type. Note that this includes the size required for padding.
   */
  uint64_t getSizeInBytes() const {
    return type_.getSizeInBytes();
  }

  //constructor for quant types
  template<size_t numSizes>
  LibTensor(dnn_lib::ElemKind elk, void * const rawdata, const std::array<dim_t, numSizes> &dims,
            const std::array<dim_t, numSizes> &pitches, const bool untouch, const float scale, const int offset)
    : ptrData_(reinterpret_cast<char*>(rawdata)),
      type_(elk, dims, pitches, scale, offset),
      untouch_(untouch) {}

  // constructor for non quant types
  template<size_t numSizes>
  LibTensor(dnn_lib::ElemKind elk, void * const rawdata, const std::array<dim_t, numSizes> &dims,
            const std::array<dim_t, numSizes> &pitches, const bool untouch)
    : ptrData_(reinterpret_cast<char*>(rawdata)),
      type_(elk, dims, pitches),
      untouch_(untouch) {}

  // constructor from type
  LibTensor(const Type &type, void * const rawdata, const bool untouch)
    : ptrData_(reinterpret_cast<char*>(rawdata)),
      type_(type),
      untouch_(untouch) {}

  LibTensor(const Type &&type, void * const rawdata, const bool untouch)
    : ptrData_(reinterpret_cast<char*>(rawdata)),
      type_(std::move(type)),
      untouch_(untouch) {}

  float getScale() const {
    return type_.getScale();
  }

  int32_t getOffset() const {
    return type_.getOffset();
  }

  size_t getElementSize() const {
    return type_.getElementSize();
  }

}; //end LibTensorBase class

}

#endif // _LIB_TENSOR_H_
