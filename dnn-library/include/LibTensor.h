/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
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
#include <numeric>
#include <string.h>
#include <tuple>
#include <type_traits>

#ifdef __riscv
#include <etsoc/isa/cacheops-umode.h>
#include <etsoc/isa/utils.h>
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
  const float scale_{};

  /*@brief On quantized tensors, this represents the offset of the values.
   */
  const int32_t offset_{};

  /*@brief When the tensor is fully broadcast, it may have a single value registered.
   */
  const bool hasSingleValue_ = false;

  /*@brief If hasSingleValue_ is true, contains the value fully broadcasted across the Tensor.
   */
  const float singleValue_ = 0.0;

  /*@brief If isCounter_ is true, the values of the tensor are contiguous with the same stride.
   */
  const bool isCounter_ = false;

  /*@brief Starting value of the counter.
   */
  const int64_t counterOffset_ = 0;

  /*@brief Stride between values of the counter.
   */
  const int64_t counterStride_ = 0;

  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  template <size_t numSizes>
  constexpr Type(const dnn_lib::ElemKind elk, const std::array<dim_t, numSizes>& dims, const float scale,
                 const int32_t offset)
    : sizes_(make_dims(dims))
    , elementType_(elk)
    , numSizes_(numSizes)
    , scale_(scale)
    , offset_(offset) {
    assert(isQuantizedElemKind(elk));
  }

  /*@brief Initialize a new non-quantized type.
   */
  template <size_t numSizes>
  constexpr Type(const dnn_lib::ElemKind elk, const std::array<dim_t, numSizes>& dims)
    : sizes_(make_dims(dims))
    , elementType_(elk)
    , numSizes_(numSizes) {
    assert(!isQuantizedElemKind(elk));
  }

  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  template <size_t numSizes>
  constexpr Type(const dnn_lib::ElemKind elk, const std::array<dim_t, numSizes>& dims,
                 const std::array<dim_t, numSizes>& strides, const float scale, const int32_t offset)
    : sizes_(make_dims(dims))
    , strides_(make_strides(strides))
    , elementType_(elk)
    , numSizes_(numSizes)
    , scale_(scale)
    , offset_(offset) {
    assert(isQuantizedElemKind(elk));
  }

  /*@brief Initialize a new non-quantized type.
   */
  template <size_t numSizes>
  constexpr Type(const dnn_lib::ElemKind elk, const std::array<dim_t, numSizes>& dims,
                 const std::array<dim_t, numSizes>& strides)
    : sizes_(make_dims(dims))
    , strides_(make_strides(strides))
    , elementType_(elk)
    , numSizes_(numSizes) {
    assert(!isQuantizedElemKind(elk));
  }

  /*@brief non templated version of the previous constructors (receiving dimensions/strides with max_tensor_dimensions,
    and an extra parameter to set the actual number of dimensions
   */
  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  constexpr Type(const dnn_lib::ElemKind elk, const size_t numSizes, const dim_array_t& dims,
                 const dim_array_t& strides, const float scale, const int32_t offset)
    : sizes_(dims)
    , strides_(strides)
    , elementType_(elk)
    , numSizes_(numSizes)
    , scale_(scale)
    , offset_(offset) {
    assert(isQuantizedElemKind(elk));
  }

  /*@brief Initialize a new non-quantized type.
   */
  constexpr Type(const dnn_lib::ElemKind elk, const size_t numSizes, const dim_array_t& dims,
                 const dim_array_t& strides)
    : sizes_(dims)
    , strides_(strides)
    , elementType_(elk)
    , numSizes_(numSizes) {
    assert(!isQuantizedElemKind(elk));
  }

  /*@brief Reshape existing type this takes care of quantized types.
   */
  template <size_t numSizes> static Type newShape(const Type& T, const std::array<dim_t, numSizes>& dims) {
    if (T.isQuantizedType())
      return Type(T.elementType_, dims, T.scale_, T.offset_);
    else
      return Type(T.elementType_, dims);
  }

  /*@brief Reshape existing type and change alignments.
   */
  template <size_t numSizes>
  static Type newShape(const Type& T, const std::array<dim_t, numSizes>& dims,
                       const std::array<dim_t, numSizes>& pitches) {
    if (T.isQuantizedType())
      return Type(T.elementType_, dims, pitches, T.scale_, T.offset_);
    else
      return Type(T.elementType_, dims, pitches);
  }

  static Type newShape(const Type& T, size_t numSizes, const dim_array_t& dims, const dim_array_t& pitches) {
    if (T.isQuantizedType())
      return Type(T.elementType_, numSizes, dims, pitches, T.scale_, T.offset_);
    else
      return Type(T.elementType_, numSizes, dims, pitches);
  }

  /*@brief Reshape existing type by taking shapes and strides of \p shapeType.
   */
  static Type newShape(const Type& kindType, const Type shapeType) {
    //@TODO  T.getElementType() == shapeType->getelementSize() Size should be the same
    if (kindType.isQuantizedType())
      return Type(kindType.elementType_, shapeType.sizes_, shapeType.strides_, kindType.scale_, kindType.offset_);
    else
      return Type(kindType.elementType_, shapeType.sizes_, shapeType.strides_);

    // TODO: the numSizes_is set wrong => because of dimension and strides extension. Either set properly (e.g. separate
    // extended
    // and non extended arrays... or maybe just delete this newShape, in case it is not needed)
  }

  /* brief returns true if \p other is the same type.
   */
  // bool isEqual(TypeRef other) const { return isEqual(*other); }

  /* brief returns true if \p other has same shape.
   */
  bool hasSameShape(const Type other) const {
    if (numSizes_ != other.getNumDims())
      return false;
    const dim_array_t& other_sizes = other.getSizes();
    const dim_array_t& other_strides = other.getStrides();
    for (size_t idx = 0; idx < numSizes_; idx++) {
      if (sizes_[idx] != other_sizes[idx])
        return false;
      if (strides_[idx] != other_strides[idx])
        return false;
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
  dim_t getNumDims() const {
    return numSizes_;
  }

  /*@brief returns the elemet type
   */
  ElemKind getElementType() const {
    return elementType_;
  }

  /*@brief return the number of elements in the tensor.
   */
  dim_t size() const {
    dim_t acum = 1;
    for (auto i : sizes_)
      acum *= i;
    return acum;
  }

  // /*@brief Calculate the size of the slice starting at \p StartDim. Returns the
  //  *number of elements in a slice in the tensor.
  //  */
  // dim_t getSliceSize(unsigned char startDim) const {

  //   assert(startDim <= numSizes_ && " Invalid start dim");
  //   dim_t s = 1;
  //   for(unsigned char i = startDim; i < numSizes_; i++) {
  //     s *= dim_t(sizes_[i]);
  //   }
  //   return s;
  // }

  /*@brief returns true if the templated parameter \p Elemkind matches this type.
   */
  template <class ElemTy> bool isType() const {
    return isType<ElemTy>(elementType_);
  }

  /*@brief returns true if the templated parameter \p ElemKind matches the type
   *that's specified by the parameter \p Ty.
   */
  template <class ElemTy> static bool isType(dnn_lib::ElemKind elk) {
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

  // /*@brief returns true if the type of this Tensor is one of the floating point
  //  *types.
  //  */
  // bool isFPType() const {
  //   return getElementType() == dnn_lib::ElemKind::FloatTy or
  //          getElementType() == dnn_lib::ElemKind::Float16Ty or
  //          getElementType() == dnn_lib::ElemKind::BFloat16Ty;
  // }

  /*@brief returns the size of the type element.
   */
  size_t getElementSize() const {
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

  /*@brief When the tensor is fully broadcast, it may have a single value registered.
   */
  bool hasSingleValue() const {
    return hasSingleValue_;
  }

  /*@brief If hasSingleValue() is true, returns the value that is repeated
   */
  float getSingleValue() const {
    assert(hasSingleValue() and "Single value can only be obtained when it is set");
    return singleValue_;
  }

  /*@brief If isCounter_ is true, the values of the tensor are contiguous with the same stride
   */
  bool isCounter() const {
    return isCounter_;
  }

  int64_t getCounterOffset() const {
    assert(isCounter() and "Counter offset can only be obtained when tensor is a counter");
    return counterOffset_;
  }

  int64_t getCounterStride() const {
    assert(isCounter() and "Counter stride can only be obtained when tensor is a counter");
    return counterStride_;
  }

  // /// Given a string \p str containing the name of an ElemKind from
  // /// Type::getElementName, returns the corresponding ElemKind or Error if a
  // /// mapping couldn't be found.
  // static dnn_lib::ElemKind getElementKindFromName(string str) {
  //   if (str == Type::getElementName(dnn_lib::ElemKind::FloatTy)) {
  //     return dnn_lib::ElemKind::FloatTy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Float16Ty)) {
  //     return dnn_lib::ElemKind::Float16Ty;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::BFloat16Ty)) {
  //     return dnn_lib::ElemKind::BFloat16Ty;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Int8QTy)) {
  //     return dnn_lib::ElemKind::Int8QTy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::UInt8QTy)) {
  //     return dnn_lib::ElemKind::UInt8QTy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Int16QTy)) {
  //     return dnn_lib::ElemKind::Int16QTy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Int32QTy)) {
  //     return dnn_lib::ElemKind::Int32QTy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Int32ITy)) {
  //     return dnn_lib::ElemKind::Int32ITy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::Int64ITy)) {
  //     return dnn_lib::ElemKind::Int64ITy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::UInt8FusedQTy)) {
  //     return dnn_lib::ElemKind::UInt8FusedQTy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::UInt8FusedFP16QTy)) {
  //     return dnn_lib::ElemKind::UInt8FusedFP16QTy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::UInt4FusedFP16QTy)) {
  //     return dnn_lib::ElemKind::UInt4FusedFP16QTy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::UInt4FusedQTy)) {
  //     return dnn_lib::ElemKind::UInt4Fused6QTy;
  //   } else if (str == Type::getElementName(dnn_lib::ElemKind::BoolTy)) {
  //     return dnn_lib::ElemKind::BoolTy;
  //   } else {
  //     //assert(true && "Invalid ElemKind string");
  //     return dnn_lib::ElemKind::FloatTy;
  //   }
  // }

}; // class Type

class LibTensor final {

private:
  char* const ptrData_;

  const Type type_;

  template <class ElemTy> friend class Handle;

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
  ElemKind getElementType() const {
    return type_.getElementType();
  }

  /*@brief returns if padding positions are untouchable.
   */
  bool getUntouchable() const {
    return untouch_;
  }

  /*@brief returns True if the coordinate is within the array.
   */
  // template<std::size_t sz>
  // bool isInBounds(std::array<dim_t, sz>& coord) const {
  //   assert(type_.numSizes_ == coord.size() && "Invalid number of indices");
  //   for (size_t i = 0u, e = coord.size(); i < e; i++) {
  //     if (coord[i] >= type_.sizes_[i]) {
  //       return false;
  //     }
  //   }
  //   return true;
  // }

  //   /*@brief set the content of the tenosr to zero. if \p resetFusedScalesOffsets,
  //    *then fused scales/offsets will be set to 1.0/0.0 as well.
  //    */
  //   void zero(bool resetFusedScalesOffsets = false) {
  //     size_t size = actualSize();
  //     //Quantized tensors should go to their offset.
  //     switch (type_.getElementType()) {
  //     case dnn_lib::ElemKind::Int8QTy: {
  //       auto *data = reinterpret_cast<int8_t *>(getData());
  //       std::fill(&data[0], &data[0] + size, (int8_t)type_.getOffset());
  //       break;
  //     }
  //     case dnn_lib::ElemKind::UInt8QTy: {
  //       auto *data = reinterpret_cast<uint8_t *>(getData());
  //       std::fill(&data[0], &data[0] + size, (uint8_t)type_.getOffset());
  //       break;
  //     }
  //     case dnn_lib::ElemKind::Int16QTy: {
  //       auto *data = reinterpret_cast<int16_t *>(getData());
  //       std::fill(&data[0], &data[0] + size, (int16_t)type_.getOffset());
  //       break;
  //     }
  //     case dnn_lib::ElemKind::Int32QTy: {
  //       auto *data = reinterpret_cast<int32_t *>(getData());
  //       std::fill(&data[0], &data[0] + size, (int32_t)type_.getOffset());
  //       break;
  //     }
  // #define FUSED_CASE(ELEM_KIND, DATA_TYPE)  case dnn_lib::ElemKind::ELEM_KIND: break
  //     /* FUSED_CASE(dnn_lib::ElemKind::UInt8FusedQTy, float);       */
  //     /* FUSED_CASE(dnn_lib::ElemKind::UInt8FusedFP16QTy, float16_t); */
  // #undef FUSED_CASE
  //      default:
  //       // Non-quantized tensors are set to 0.
  //        for(dim_t i = 0; i < (size * type_.getElementSize()); i++) {

  //        }

  //       break;
  //     }

  //   }

  /*@brief Get number of dimensions the tensor has
   */
  dim_t ndims() const {
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
  dim_array_t stridesNoPadding() const {
    dim_array_t v;
    v[ndims() - 1] = 1;
    for (sdim_t i = ndims() - 2; i >= 0; i--) {
      v[i] = v[i + 1] * dims()[i + 1];
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

  bool hasSingleValue() const {
    return type_.hasSingleValue();
  }

  /*@brief If hasSingleValue() is true, returns the value that is repeated
   */
  float getSingleValue() const {
    return type_.getSingleValue();
  }

  /*@brief If isCounter_ is true, the values of the tensor are contiguous with the same stride.
   */
  bool isCounter() const {
    return type_.isCounter();
  }

  /*@brief Starting value of the counter.
   */
  int64_t getCounterOffset() const {
    return type_.getCounterOffset();
  }

  /*@brief Stride between values of the counter.
   */
  int64_t getCounterStride() const {
    return type_.getCounterStride();
  }

  // constructor for quant types
  template <size_t numSizes>
  LibTensor(dnn_lib::ElemKind elk, void* const rawdata, const std::array<dim_t, numSizes>& dims,
            const std::array<dim_t, numSizes>& pitches, const bool untouch, const float scale, const int offset)
    : ptrData_(reinterpret_cast<char*>(rawdata))
    , type_(elk, dims, pitches, scale, offset)
    , untouch_(untouch) {
  }

  // constructor for non quant types
  template <size_t numSizes>
  LibTensor(dnn_lib::ElemKind elk, void* const rawdata, const std::array<dim_t, numSizes>& dims,
            const std::array<dim_t, numSizes>& pitches, const bool untouch)
    : ptrData_(reinterpret_cast<char*>(rawdata))
    , type_(elk, dims, pitches)
    , untouch_(untouch) {
  }

  // constructor from type
  LibTensor(const Type& type, void* const rawdata, const bool untouch)
    : ptrData_(reinterpret_cast<char*>(rawdata))
    , type_(type)
    , untouch_(untouch) {
  }

  LibTensor(const Type&& type, void* const rawdata, const bool untouch)
    : ptrData_(reinterpret_cast<char*>(rawdata))
    , type_(std::move(type))
    , untouch_(untouch) {
  }

  // LibTensor(const LibTensor &other) = delete;
  // LibTensor &operator=(const LibTensor &other) = delete;

  /*@brief return a new handle that points and manages this tensor.
   */

  template <class ElemTy> Handle<ElemTy> getHandle() & {
    //@TODO check Elemty type_.isType<ElemType>() handle to wrong type.
    return Handle<ElemTy>(this);
  }

  template <class ElemTy> const Handle<ElemTy> getHandle() const& {
    //@TODO check Elemty type_.isType<ElemType>() handle to wrong type.
    return Handle<ElemTy>(const_cast<LibTensor*>(this));
  }

  template <class ElemTy = float> Handle<ElemTy> getHandle() && = delete;

  /* @brief copy raw data value at ptrData_ buffer tensor given at \p inTensor
   * to the other ptrData_ buffer at (this).
   *
   * @param[in] inTensor tensor from copy data
   */
  void copyRawFrom(LibTensor* inT) {
    //@TODO check both tensor has the same shape!!
    memcpy(ptrData_, inT->ptrData_, inT->getSizeInBytes());
  }

  // /*@brief create a new copy of the current tensor.
  //  */
  // LibTensor clone() const {
  //   LibTensor slice;
  //      slice.assign(this);
  //   return slice;
  // }

  /*@brief return the raw unsafe pointer to the tensor payload.
   */
  // TODO: REMOVE if not used  char* getUnsafePtr() const { return getData(); }

  /*TODO: After re-do sw-2429 (refact operands) are the getters necessary? if not remove them. */
public:
  float getScale() const {
    return type_.getScale();
  }
  int32_t getOffset() const {
    return type_.getOffset();
  }
  size_t getElementSize() const {
    return type_.getElementSize();
  }

  /*@brief returns a pointer to the raw data, of type \p ElemTy.
   */
  template <class ElemTy> ElemTy* getRawDataPointer() {
    //@TODO check Elemty is type_.isType<>()
    constexpr size_t alignment = alignof(ElemTy);
    assert(uintptr_t(ptrData_) % alignment == 0);
    return reinterpret_cast<ElemTy*>(__builtin_assume_aligned(ptrData_, alignment));
  }

  void* getRawDataPointer() {
    return reinterpret_cast<void*>(ptrData_);
  }

  /*@brief returns a const pointer to the raw data, of type \p ElemTy.
   */
  template <class ElemTy> const ElemTy* getRawDataPointer() const {
    constexpr size_t alignment = alignof(ElemTy);
    assert(uintptr_t(ptrData_) % alignment == 0);
    return reinterpret_cast<ElemTy*>(__builtin_assume_aligned(ptrData_, alignment));
  }

  const void* getRawDataPointer() const {
    return reinterpret_cast<void*>(ptrData_);
  }

  // returns offset and maxRead (in number of elements)
  void partitionCL(const size_t minionId, const size_t activeMinions, dim_t& offset, dim_t& maxRead) const {

    size_t onlyMin0;   // elements only for minion0 (e.g. unaligned bytes)
    size_t firstSpare; // first minion with 1 CL less
    size_t CLperMin;   // CL to process per minion
    size_t elementSize = getElementSize();

    // if less that 1 CL... all to min0
    if (getSizeInBytes() <= CACHE_LINE_BYTES) {
      onlyMin0 = getSizeInBytes();
      firstSpare = 0;
      CLperMin = 0;
    } else {
      // get number of non aligned elements, and subtract from the total size
      // these unaligned elements will be processed by minion 0
      onlyMin0 = (CACHE_LINE_BYTES - reinterpret_cast<size_t>(getAddress()) % CACHE_LINE_BYTES) % CACHE_LINE_BYTES;
      int64_t aligned = getSizeInBytes() - onlyMin0;

      // total number of involved CL
      size_t inCL = (aligned + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
      // CL per minion
      CLperMin = inCL / activeMinions;
      // and minions with 1 CL lss
      firstSpare = inCL - CLperMin * activeMinions;
    }

#if 1
    if (minionId < firstSpare) {
      offset = (CLperMin + 1) * minionId;
      maxRead = CLperMin + 1;
    } else {
      offset = CLperMin * minionId + firstSpare;
      maxRead = CLperMin;
    }
#else // same as above, but avoiding branches
    // mask is 0 if minionId >= firstSpare, 0xffff...fff (-1) if minionId < firstSpare
    uint64_t mask = static_cast<int64_t>(minionId - firstSpare) >> 63;
    offset = CLperMin * minionId + (minionId & mask) + (~mask & firstSpare);
    maxRead = CLperMin - mask;
#endif

    offset *= CACHE_LINE_BYTES / elementSize;
    maxRead *= CACHE_LINE_BYTES / elementSize;

    if (minionId == 0) {
      maxRead += onlyMin0 / elementSize;
    } else {
      offset += onlyMin0 / elementSize;
    }

    // Prevents going beyond the tensor limits (this can actually happen for the
    // last minion) and overwrite other valid data
    maxRead = std::min<dim_t>(maxRead, (type_.getSizeInBytes() / elementSize) - offset);
  }

  dim_array_t offset2Coord(size_t offset) const {
    dim_array_t coords = {0};
    uint32_t rm = static_cast<uint32_t>(offset); // operations in uint32_t.. division is faster
    for (size_t i = 0; i < ndims(); i++) {
      coords[i] = rm / static_cast<uint32_t>(strides()[i]);
      rm = rm - static_cast<uint32_t>(coords[i]) * static_cast<uint32_t>(strides()[i]);
    }
    return coords;
  }

  void evict(uint64_t dst, size_t offset, size_t count) const {
#ifdef __riscv
    FENCE;
    const size_t typeSize = getElementSize();
    size_t cl = (count * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
    assert(cl > 0);
    uintptr_t addr = reinterpret_cast<uintptr_t>(getAddress()) + typeSize * offset;
    while (cl > 16) {
      cache_ops_evict_va(0, dst, addr, 15, CACHE_LINE_BYTES, 0);
      addr += (CACHE_LINE_BYTES * 16);
      cl -= 16;
    }
    if (cl > 0)
      cache_ops_evict_va(0, dst, addr, cl - 1, CACHE_LINE_BYTES, 0);
#else
    assert("shouldn't call this function unless it is from minion");
#endif
  }

  void evict(uint64_t dst) const {
    evict(dst, 0, actualSize());
  }

  /* partitions current tensor in cache lines, and loops over all elements
     in steps of 'step' elements and skipping padding, calling the 'compute'
     function.

     The compute function has to be of the form:
       compute( uintptr_t outPtr, uint_ptr inPtr, dim_t valid, computeArgs...)
     where:
        outPtr = pointer to the element to write ( *this tensor)
        inPtr  = pointer to the element to read
        valid  = number of valid elements in the current row starting at outPtr
        computeArgs = any extra parameters that will just be forwared

  */
  template <typename dstType, typename srcType = dstType, dim_t step = 8, bool doEvicts = true, typename compute_t,
            typename... computeArgs_t>
  INLINE_ATTR void partitionLoop(const size_t minionId, const size_t activeMinions, const uint64_t flags,
                                 LibTensor* inT, compute_t compute, computeArgs_t&&... computeArgs) {

    ////////////////////////////////////////////////////////////////////////////////
    // partition work between minions in multiples of CL
    ////////////////////////////////////////////////////////////////////////////////
    dim_t first; // first element in raw array to process
    dim_t count; // nr  elements in to process (will be in multiples of CL)

    partitionCL(minionId, activeMinions, first, count);
    if (unlikely(count == 0))
      return; // minion has no work to do

    ////////////////////////////////////////////////////////////////////////////////
    // and loop
    ////////////////////////////////////////////////////////////////////////////////

    // setup
    const auto N = ndims();
    const dim_t lastDim = dims()[N - 1];

    const dim_t endOffset = first + count;

    const srcType* const srcP = inT->getRawDataPointer<srcType>();
    const dim_t srcLastPitch = inT->strides()[N - 1];
    dstType* const dstP = getRawDataPointer<dstType>();
    const dim_t dstLastPitch = strides()[N - 1];

    bool consecutiveFeatures = (dstLastPitch == 1) and (srcLastPitch == 1);
    dim_t iterationStep =
      (consecutiveFeatures ? step : 1); // When features are not consecutive, we need to operate element by element

    ////////////////////////////////////////////////////////////////////////////////
    // simpler loop if there is just one dimension
    ////////////////////////////////////////////////////////////////////////////////
    if ((N == 1) and consecutiveFeatures) {
      // simplification: just 1 loop, and offset in elements is the same for all the tensors
      for (dim_t offset = first; offset < endOffset; offset += step) {
        dim_t elems = std::min<dim_t>(step, lastDim - offset);
        elems = std::min<dim_t>(elems, endOffset - offset);
        compute(reinterpret_cast<uintptr_t>(dstP + offset), reinterpret_cast<uintptr_t>(srcP + offset), elems,
                std::forward<computeArgs_t>(computeArgs)...);
      }
    } else {
      ////////////////////////////////////////////////////////////////////////////////
      // Loops for more than 1 dim
      ////////////////////////////////////////////////////////////////////////////////
      // get iterators to loop through all the dimensions
      auto out = getHandle<dstType>().getIterator(first);
      auto in = inT->getHandle<srcType>().getIterator(out.coords());

      for (dim_t oOffset = out.offset(); oOffset < endOffset; oOffset = out.offset()) {
        dim_t iOffset = in.offset();
        size_t stop = std::min(lastDim - out.coords()[N - 1], endOffset - oOffset);
        dim_t elems = std::min<dim_t>(iterationStep, stop);
        compute(reinterpret_cast<uintptr_t>(dstP + oOffset), reinterpret_cast<uintptr_t>(srcP + iOffset), elems,
                std::forward<computeArgs_t>(computeArgs)...);
        out += elems;
        in += elems;
      }
    }

    evict(DO_EVICTS, first, count);
  }

  /* same as above, but generalized to N input tensors, each with different types
    partitions current tensor in cache lines, and loops over all elements
   in steps of 'step' elements and skipping padding, calling the 'compute'
   function.

   The compute function has to be of the form:
     compute( uintptr_t outPtr, uint_ptr inPtr0,[ uint_ptr inPtr1...,] dim_t valid, computeArgs...)
   where:
      outPtr = pointer to the element to write ( *this tensor)
      inPtrX  = pointer to the elements to read
      valid  = number of valid elements in the current row starting at outPtr
      computeArgs = any extra parameters that will just be forwared

   The number of tensors and its type depend on the template parameters that should be auto deduced
 */
  template <typename dstType, dim_t step = 8, bool doEvicts = true, typename compute_t, typename... computeArgs_t,
            typename... srcTypes, typename... tensorTypes, size_t... idx>
  INLINE_ATTR void partitionLoop(const size_t minionId, const size_t activeMinions, const uint64_t flags,
                                 const std::tuple<tensorTypes*...>& inT, const std::tuple<srcTypes...>&,
                                 const std::index_sequence<idx...>&, compute_t compute,
                                 computeArgs_t&&... computeArgs) {

    static_assert(sizeof...(srcTypes) == sizeof...(tensorTypes) && sizeof...(idx) == sizeof...(tensorTypes),
                  "number of src types and parameters has to match!");
    constexpr size_t nrInputs = sizeof...(srcTypes);
    ////////////////////////////////////////////////////////////////////////////////
    // partition work between minions in multiples of CL
    ////////////////////////////////////////////////////////////////////////////////
    dim_t first; // first element in raw array to process
    dim_t count; // nr  elements in to process (will be in multiples of CL)

    partitionCL(minionId, activeMinions, first, count);
    if (unlikely(count == 0))
      return; // minion has no work to do

    ////////////////////////////////////////////////////////////////////////////////
    // and loop
    ////////////////////////////////////////////////////////////////////////////////

    // setup
    const auto N = ndims();
    const dim_t lastDim = dims()[N - 1];
    const dim_t endOffset = first + count;

    const auto srcPs = std::make_tuple(std::get<idx>(inT)->template getRawDataPointer<srcTypes>()...);
    const auto srcLastPitches = std::make_tuple(std::get<idx>(inT)->strides()[N - 1]...);
    dstType* const dstP = getRawDataPointer<dstType>();
    const auto dstLastPitch = strides()[N - 1];

    bool consecutiveFeatures = (dstLastPitch == 1) and ((std::get<idx>(srcLastPitches) == 1) and ...);
    dim_t iterationStep =
      (consecutiveFeatures ? step : 1); // When features are not consecutive, we need to operate element by element

    ////////////////////////////////////////////////////////////////////////////////
    // simpler loop if there is just one dimension
    ////////////////////////////////////////////////////////////////////////////////
    if ((N == 1) and consecutiveFeatures) {
      // simplification: just 1 loop, and offset in elements is the same for all the tensors
      for (dim_t offset = first; offset < endOffset; offset += step) {
        dim_t elems = std::min<dim_t>(step, lastDim - offset);
        elems = std::min<dim_t>(elems, endOffset - offset);
        compute(reinterpret_cast<uintptr_t>(dstP + offset),
                reinterpret_cast<uintptr_t>(std::get<idx>(srcPs) + offset)..., elems,
                std::forward<computeArgs_t>(computeArgs)...);
      }
    } else {
      ////////////////////////////////////////////////////////////////////////////////
      // Loops for more than 1 dim
      ////////////////////////////////////////////////////////////////////////////////

      // get iterators to loop through all the dimensions
      auto out = getHandle<dstType>().getIterator(first);
      auto in = std::make_tuple(std::get<idx>(inT)->template getHandle<srcTypes>().getIterator(out.coords())...);

      for (dim_t oOffset = out.offset(); oOffset < endOffset; oOffset = out.offset()) {
        std::array<dim_t, nrInputs> iOffsets = {(std::get<idx>(in)).offset()...};
        size_t stop = std::min(lastDim - out.coords()[N - 1], endOffset - oOffset);
        dim_t elems = std::min<dim_t>(iterationStep, stop);
        compute(reinterpret_cast<uintptr_t>(dstP + oOffset),
                reinterpret_cast<uintptr_t>(std::get<idx>(srcPs) + iOffsets[idx])..., elems,
                std::forward<computeArgs_t>(computeArgs)...);
        out += elems;
        (std::get<idx>(in).operator+=(elems), ...);
      }
    }

    evict(DO_EVICTS, first, count);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // simpler interfaces to the generic partitionLoop for 2 elements
  ////////////////////////////////////////////////////////////////////////////////

  template <typename dstType, typename srcType1 = dstType, typename srcType2 = dstType, dim_t step = 8,
            bool doEvicts = true, typename compute_t, typename... computeArgs_t>
  INLINE_ATTR void partitionLoop2(const size_t minionId, const size_t activeMinions, const uint64_t flags,
                                  LibTensor* in1T, LibTensor* in2T, compute_t compute, computeArgs_t&&... computeArgs) {

    partitionLoop<dstType, step, doEvicts, compute_t, computeArgs_t...>(
      minionId, activeMinions, flags, std::make_tuple(in1T, in2T), std::tuple<srcType1, srcType2>{},
      std::make_index_sequence<2>{}, compute, std::forward<computeArgs_t>(computeArgs)...);
  }

}; // end LibTensorBase class

/*@brief Convert to flattened 1d offset given \p indices.
 *
 *@param[inout] indices keeps the coords of the element
 *@return
 */
template <size_t N>
INLINE_ATTR size_t getFlattenedOffset(const std::array<dim_t, N>& indices, const dim_array_t& strides) {
  /*@TODO check indices size isn't bigger than strides*/
  // assert(indices.size() <= strides.size());
  size_t r = 0;
  for (size_t i = 0; i < N; i++)
    r += indices[i] * strides[i];
  return r;
}

template <size_t N>
INLINE_ATTR size_t getFlattenedOffset(const std::array<dim_t, N>& indices, const dim_array_t& strides,
                                      const dim_array_t& extStrides, size_t ndx) {

  size_t r = 0;
  for (size_t i = 0; i < N; i++) {
    if (i == ndx)
      r += indices[i] * extStrides[i];
    else
      r += indices[i] * strides[i];
  }
  return r;
}

#include "LibTensorIterator.h"

template <class ElemTy> class Handle final {

  /*brief pointer to the tensor that this handle wraps.
   */
  LibTensor* const tensor_;

  /*@brief It has the mult of the sizes for each position to end.
   */
  const dim_array_t& strides_ __attribute__((aligned(8)));

  const dim_array_t& sizes_ __attribute__((aligned(8)));

  /*@brief the number of dimensions used in the tensor.
   */
  const dim_t numDims_;

public:
  using iterator = HandleIterator<ElemTy>;
  friend class HandleIterator<ElemTy>;

  const iterator begin() {
    return iterator(*this);
  }
  const iterator end() {
    return iterator(*this, sizes_[0] * strides_[0], sizes_);
  }
  iterator getIterator(size_t offset) {
    return iterator(*this, offset);
  }
  iterator getIterator(const dim_array_t& coords) {
    return iterator(*this, coords);
  }

  /*@brief Calculate the index for a specific element in the tensor.
   *
   *@param[inout] coords indices to access element. It has to have the same
   * dimensions as tensor to be acessed.
   *@return flattened 1D element position.
   */
  template <size_t N> size_t getElementPtr(const std::array<dim_t, N>& indices) const {
    return getFlattenedOffset(indices, strides_);
  }

  template <size_t N>
  size_t getElementPtr(const std::array<dim_t, N>& indices, const dim_array_t& extStrides, size_t ndx) const {
    return getFlattenedOffset(indices, strides_, extStrides, ndx);
  }

  /*@brief returns the value of the n'th dimension \p dim, for the index \p idx.
   * 0 <= idx < size(), meaning that \p idx addresses a real data elements,
   * not paddings.
   */
  size_t getDimForPtr(size_t dim, size_t idx) const {
    // assert(dim < numDims_ && "Invalid dimension");
    // assert(idx < size() && "Invalid index");
    auto R = idx;
    for (size_t i = dim + 1; i < numDims_; i++) {
      R /= sizes_[i];
    }
    return R % sizes_[dim];
  }

  /*@brief returns the type of the tensor.
   */
  const Type& getType() const {
    return tensor_->getType();
  }

  /*@brief returns the element type of the tensor.
   */
  ElemKind getElementType() const {
    return tensor_->getElementType();
  }

  /*@brief Construct a Tensor handle.
   */
  Handle(LibTensor* tensor)
    : tensor_(tensor)
    , strides_(tensor->strides())
    , sizes_(tensor->dims())
    , numDims_(tensor->ndims()) {
  }

  /*@brief returns the number of elements in the whole tensor.
   */
  dim_t size() const {
    return tensor_->size();
  }

  /*@brief returns the number of elements in the tensor taking striding/pitches
   *into account. Since size() does not take striding into account, size() is
   *always <= actualSize():
   */
  dim_t actualSize() const {
    return tensor_->actualSize();
  }

  /*@brief check if given \p indices is into the dims_ bounds.
   *
   *@param[inout] indices
   *@return true if indices is into the bounds.
   */
  /* bool isInBounds(dim_t* indices) const { */
  /*   return tensor_->isInBounds(indices); */
  /* } */
  // template<std::size_t sz>
  // bool isInBounds(std::array<dim_t, sz>& indices) const {
  //   return tensor_->isInBounds(indices);
  // }

  void clear(ElemTy value = 0) {
    std::fill(this->begin(), this->end(), value);
  }

  void zero(void) {
    clear(0);
  }
  /*@brief return reference to a meaningful data element. This method skip
   *padding elements.
   */
  template <size_t N> ElemTy& at(std::array<dim_t, N> indices) {
    size_t index = getElementPtr(indices);
    auto* data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  template <size_t N> const ElemTy& at(std::array<dim_t, N> indices) const {
    size_t index = getElementPtr(indices);
    auto* data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /*@brief specific case use strides from outside of tensor */
  template <size_t N> ElemTy& at(std::array<dim_t, N> indices, const dim_array_t& extStrides, size_t ndx) {
    size_t index = getElementPtr(indices, extStrides, ndx);
    auto* data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /*@brief returns the element at position offset \p idx without any size
   * of calculation. The returned element can be a pad element.*/
  ElemTy& raw(size_t index) {
    auto* data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  /*@brief returns the element at position offset \p idx without any size
   * of calculation. The returned element can be a pad element.*/
  const ElemTy& raw(size_t index) const {
    auto* data = tensor_->getRawDataPointer<ElemTy>();
    return data[index];
  }

  float getScale(void) {
    return tensor_->getScale();
  }
  int32_t getOffset(void) {
    return tensor_->getOffset();
  }

  float getScale(void) const {
    return tensor_->getScale();
  }
  int32_t getOffset(void) const {
    return tensor_->getOffset();
  }

}; // end Handle class

// returns offset and maxRead (in number of elements)
template <size_t bytesPerElement>
INLINE_ATTR void partitionCL(uintptr_t address, dim_t sizeInBytes, const uint64_t minionId, const size_t activeMinions,
                             dim_t& offset, dim_t& maxRead) {

  size_t onlyMin0;   // elements only for minion0 (e.g. unaligned bytes)
  size_t firstSpare; // first minion with 1 CL less
  size_t CLperMin;   // CL to process per minion

  // if less that 1 CL... all to min0
  if (sizeInBytes <= CACHE_LINE_BYTES) {
    onlyMin0 = sizeInBytes;
    firstSpare = 0;
    CLperMin = 0;
  } else {
    // get number of non aligned elements, and subtract from the total size
    // these unaligned elements will be processed by minion 0
    onlyMin0 = (CACHE_LINE_BYTES - static_cast<size_t>(address) % CACHE_LINE_BYTES) % CACHE_LINE_BYTES;
    int64_t aligned = sizeInBytes - onlyMin0;

    // total number of involved CL
    size_t inCL = (aligned + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
    // CL per minion
    CLperMin = inCL / activeMinions;
    // and minions with 1 CL lss
    firstSpare = inCL - CLperMin * activeMinions;
  }

#if 1
  if (minionId < firstSpare) {
    offset = (CLperMin + 1) * minionId;
    maxRead = CLperMin + 1;
  } else {
    offset = CLperMin * minionId + firstSpare;
    maxRead = CLperMin;
  }
#else // same as above, but avoiding branches
  // mask is 0 if minionId >= firstSpare, 0xffff...fff (-1) if minionId < firstSpare
  uint64_t mask = static_cast<int64_t>(minionId - firstSpare) >> 63;
  offset = CLperMin * minionId + (minionId & mask) + (~mask & firstSpare);
  maxRead = CLperMin - mask;
#endif

  offset *= CACHE_LINE_BYTES / bytesPerElement;
  maxRead *= CACHE_LINE_BYTES / bytesPerElement;

  if (minionId == 0) {
    maxRead += onlyMin0 / bytesPerElement;
  } else {
    offset += onlyMin0 / bytesPerElement;
  }

  // Prevents going beyond the tensor limits (this can actually happen for the
  // last minion) and overwrite other valid data
  maxRead = std::min<dim_t>(maxRead, (sizeInBytes / bytesPerElement) - offset);
}

template <size_t bytesPerElement>
INLINE_ATTR void evict(uintptr_t address, uint64_t dst, size_t offset, size_t allocationSizeElements) {
  FENCE;
  size_t allocationBytes = allocationSizeElements * bytesPerElement;
  size_t cl = (allocationBytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  assert(cl > 0);
  uintptr_t addr = address + bytesPerElement * offset;
  while (cl > 16) {
    cache_ops_evict_va(0, dst, addr, 15, CACHE_LINE_BYTES, 0);
    addr += (CACHE_LINE_BYTES * 16);
    cl -= 16;
  }
  if (cl > 0)
    cache_ops_evict_va(0, dst, addr, cl - 1, CACHE_LINE_BYTES, 0);
}

template <size_t bytesPerElement>
INLINE_ATTR void evict(uintptr_t address, size_t allocationSizeElements, uint64_t dst) {
  evict<bytesPerElement>(address, dst, 0, allocationSizeElements);
}

#define DISPATCHER(TEMPL_ARGS, NAME, FUNCTOR)                                                                          \
  template <TEMPL_ARGS typename dstTensorType, typename dstType, typename... srcTensorTypes, typename... srcTypes,     \
            size_t... idx, typename... computeArgs_t>                                                                  \
  INLINE_ATTR void NAME(const size_t minionId, const size_t activeMinions, const uint64_t flags,                       \
                        dstTensorType* outTensor, const dstType&, const std::tuple<srcTensorTypes*...>& inTensors,     \
                        const std::tuple<srcTypes...>&, const std::index_sequence<idx...>&,                            \
                        computeArgs_t&&... computeArgs) {                                                              \
                                                                                                                       \
    constexpr size_t step = 8;                                                                                         \
    constexpr size_t nrInputs = sizeof...(srcTensorTypes);                                                             \
                                                                                                                       \
    static_assert(nrInputs == sizeof...(srcTypes) && nrInputs == sizeof...(idx));                                      \
                                                                                                                       \
    dim_t first; /* first element in raw array to process*/                                                            \
    dim_t count; /* nr  elements in to process (will be in multiples of CL) */                                         \
    const uintptr_t address = reinterpret_cast<uintptr_t>(outTensor->getAddress());                                    \
    const dim_t sizeInBytes = outTensor->getSizeInBytes();                                                             \
    partitionCL<sizeof(dstType)>(address, sizeInBytes, minionId, activeMinions, first, count);                         \
                                                                                                                       \
    if (unlikely(count == 0))                                                                                          \
      return; /* minion has no work to do*/                                                                            \
                                                                                                                       \
    /* setup */                                                                                                        \
    const dim_t N = outTensor->ndims();                                                                                \
                                                                                                                       \
    const dim_t lastDim = outTensor->dims()[N - 1];                                                                    \
    const dim_t endOffset = first + count;                                                                             \
    const auto srcPs = std::make_tuple(std::get<idx>(inTensors)->template getRawDataPointer<srcTypes>()...);           \
    auto* const dstP = outTensor->template getRawDataPointer<dstType>();                                               \
                                                                                                                       \
    if (N == 1) {                                                                                                      \
      /* Just 1 dim, and offset in elements is the same for all the tensors*/                                          \
      dim_t offset = first;                                                                                            \
      while (offset < endOffset) {                                                                                     \
        dim_t elems = step;                                                                                            \
        elems = std::min<dim_t>(elems, lastDim - offset);                                                              \
        elems = std::min<dim_t>(elems, endOffset - offset);                                                            \
        FUNCTOR(reinterpret_cast<uintptr_t>(dstP + offset),                                                            \
                reinterpret_cast<uintptr_t>(std::get<idx>(srcPs) + offset)..., elems, computeArgs...);                 \
        offset += step;                                                                                                \
      }                                                                                                                \
    } else {                                                                                                           \
      /* Loop for more than 1 dim */                                                                                   \
                                                                                                                       \
      /* get iterators to loop through all the dimensions except the last one*/                                        \
      auto out = outTensor->template getHandle<dstType>().getIterator(first);                                          \
      dim_t oOffset = out.offset();                                                                                    \
      auto in =                                                                                                        \
        std::make_tuple(std::get<idx>(inTensors)->template getHandle<srcTypes>().getIterator(out.coords())...);        \
      std::array<dim_t, nrInputs> iOffsets = {(std::get<idx>(in)).offset()...};                                        \
                                                                                                                       \
      if (out.coords()[N - 1] != 0) {                                                                                  \
        /* First iterate until completing the first feature dimension (in case initial coordinates are in the middle   \
         * of*/                                                                                                        \
        /* the row */                                                                                                  \
        size_t stop = std::min(lastDim - out.coords()[N - 1], endOffset - oOffset);                                    \
        for (size_t i = 0; i < stop; i += step) {                                                                      \
          /* Clips min values */                                                                                       \
          dim_t elems = std::min<dim_t>(step, stop - i);                                                               \
          FUNCTOR(reinterpret_cast<uintptr_t>(dstP + (oOffset + i)),                                                   \
                  reinterpret_cast<uintptr_t>(std::get<idx>(srcPs) + (iOffsets[idx] + i))..., elems, computeArgs...);  \
        }                                                                                                              \
        out += stop;                                                                                                   \
        (std::get<idx>(in).operator+=(stop), ...);                                                                     \
      }                                                                                                                \
                                                                                                                       \
      /* Then, complete the remaining iterations*/                                                                     \
      for (; out.offset() < endOffset;                                                                                 \
           out.step(N - 2), (std::get<idx>(in).step(N - 2), ...)) { /* step 2n outer dimension */                      \
        assume(out.coords()[N - 1] == 0);                                                                              \
                                                                                                                       \
        oOffset = out.offset();                                                                                        \
        iOffsets = {(std::get<idx>(in)).offset()...};                                                                  \
        size_t stop = std::min(lastDim, endOffset - oOffset);                                                          \
        for (size_t i = 0; i < stop; i += step) {                                                                      \
          /* Clips min values */                                                                                       \
          dim_t elems = std::min<dim_t>(step, stop - i);                                                               \
          FUNCTOR(reinterpret_cast<uintptr_t>(dstP + (oOffset + i)),                                                   \
                  reinterpret_cast<uintptr_t>(std::get<idx>(srcPs) + (iOffsets[idx] + i))..., elems, computeArgs...);  \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
                                                                                                                       \
    evict<sizeof(dstType)>(DO_EVICTS, first, count);                                                                   \
  }

} // namespace dnn_lib

#endif // _LIB_TENSOR_H_
