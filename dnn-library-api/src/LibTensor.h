/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef LIB_TENSOR_H
#define LIB_TENSOR_H

// Local
#include "dnnLibraryApi/LibTypes.h"

// STD
#include <cassert>
#include <cstring>
#include <numeric>
#include <tuple>

namespace dnn_lib {

class Type {
public:
  /*@brief Initialize a new type with broadcast information
   */
  Type(const dnn_lib::ElemKind elk, const size_t numSizes, const dim_array_t& dims, const dim_array_t& strides,
       const float scale, const int32_t offset, const bool hasSingleValue, const float singleValue,
       const bool isCounter, const int64_t counterOffset, const int64_t counterStride);

  /*@brief Initialize a new quantized type with \p scale an \p offset.
   */
  Type(const dnn_lib::ElemKind elk, const size_t numSizes, const dim_array_t& dims, const dim_array_t& strides,
       const float scale, const int32_t offset);

  /*@brief Initialize a new non-quantized type.
   */
  Type(const dnn_lib::ElemKind elk, const size_t numSizes, const dim_array_t& dims, const dim_array_t& strides);

  /* brief returns true if \p other has same shape.
   */
  bool hasSameShape(const Type other) const;

  /*@brief returns the scale of a quantized type.
   */
  float getScale() const;

  /*@brief returns the offset of quantized type.
   */
  int32_t getOffset() const;

  /*@brief returns the Tensor sizes_.
   */
  const dim_array_t& getSizes() const;

  /*@brief returns the Tensor strides_.
   */
  const dim_array_t& getStrides() const;

  /*@brief returns the Tensor dimension.
   */
  dim_t getNumDims() const;

  /*@brief returns the elemet type
   */
  ElemKind getElementType() const;

  /*@brief return the number of elements in the tensor.
   */
  const dim_t size() const;

  /*@brief returns true if the templated parameter \p Elemkind matches this type.
   */
  template <class ElemTy> bool isType() const {
    return isType<ElemTy>(elementType_);
  }

  /*@brief When the tensor is fully broadcast, it may have a single value registered.
   */
  bool hasSingleValue() const;

  /*@brief If hasSingleValue_ is true, contains the value fully broadcasted across the Tensor
   */
  float getSingleValue() const;

  /*@brief When the tensor is fully broadcast, it may have a single value registered.
   */
  bool isCounter() const;

  int64_t getCounterOffset() const;

  int64_t getCounterStride() const;

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
  bool isQuantizedType() const;

  /*@brief true if the type of this Tensor is one of the and index type
   */
  bool isIndexType() const;

  /*@brief returns the size of the type element.
   */
  unsigned getElementSize() const;

  /*@brief returns the size in bytes for this Tensor.
   */
  size_t getSizeInBytes() const;

  /*@brief the actual number of elements in the tensor taking striding into
   * account. Since size() does not take striding into account, size() is
   * always <= actualSize()
   */
  size_t actualSize() const;

  bool isQuantizedElemKind() const;
  bool isIndexElemKind() const;

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

private:
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

  /*@brief If hasSingleValue_ is true, contains the value fully broadcasted across the Tensor
   */
  const float singleValue_ = 0.0;

  const bool isCounter_ = false;

  const int64_t counterOffset_ = 0;

  const int64_t counterStride_ = 0;

}; // class Type

class LibTensor final {
private:
  char* const ptrData_;
  const Type type_;
  const bool untouch_;

public:
  /* @brief returns the start address of the tensor.
   */
  char* getAddress() const;

  /* @brief returns the type of the tensor.
   */
  const Type& getType() const;

  /*@brief returns the element type of the tensor.
   */
  const ElemKind getElementType() const;

  /*@brief Get number of dimensions the tensor has
   */
  const dim_t ndims() const;

  /*@brief returns the dimensions (padded with 1 until max_tensor_dimensions)
   */
  const dim_array_t& dims() const;

  /*@brief returns the strides (padded with 0 until max_tensor_dimensions)
   */
  const dim_array_t& strides() const;

  /*@brief returns strides as if there were no padding
   */
  const dim_array_t stridesNoPadding() const;

  /*@brief returns the number of real menaingful elements in the tensor. Does
   *not take strides into account.
   */
  dim_t size() const;

  /*@brief returns the actaul number of elements in the tensor taking stridding
   *into account. Since size() does not take striding into account, size() is
   *always <= actualSize(),
   */
  dim_t actualSize() const;

  /*@brief returns the number of bytes required to store the tensor based on its
   *Type. Note that this includes the size required for padding.
   */
  uint64_t getSizeInBytes() const;

  // constructor from type
  LibTensor(const Type& type, void* const rawdata, const bool untouch);

  LibTensor(const Type&& type, void* const rawdata, const bool untouch);

  LibTensor(const Tensor& tensor);

  float getScale() const;

  int32_t getOffset() const;

  size_t getElementSize() const;

  bool getUntouchable() const;

  bool hasSingleValue() const;

  float getSingleValue() const;

  bool isCounter() const;

  int64_t getCounterOffset() const;

  int64_t getCounterStride() const;
};

} // namespace dnn_lib

#endif // _LIB_TENSOR_H_
