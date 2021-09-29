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

#ifndef LIB_TYPES_H
#define LIB_TYPES_H

#include <array>
#include <stdint.h>

//#define DIM_T_32
namespace dnn_lib {

#define CACHE_LINE_BYTES 64

#ifdef DIM_T_32
using dim_t = uint32_t;
using sdim_t = int32_t;
#else
using dim_t = size_t;
using sdim_t = int64_t;
#endif

constexpr unsigned max_tensor_dimensions = 6;

using dim_array_t = std::array<dim_t, max_tensor_dimensions>;
using sdim_array_t = std::array<sdim_t, max_tensor_dimensions>;

/// An enum representing the type used by the elements of a tensor. The types of
/// Handles for these tensors should match the element kind.
enum ElemKind : unsigned char {
  // 32-bit float type (float)
  FloatTy,
  // 16-bit float type (half, fp16)
  Float16Ty,
  // 16-bit float type (bfloat16)
  BFloat16Ty,
  // 8-bit quantized type (int8_t)
  Int8QTy,
  // unsigned 8-bit quantized type (uint8_t)
  UInt8QTy,
  // 16-bit quantized type (int16_t)
  Int16QTy,
  // 32-bit quantized type (int32_t)
  Int32QTy,
  // 32-bit index type (int32_t)
  Int32ITy,
  // 64-bit index type (int64_t)
  Int64ITy,
  // 8-bit quantized type with fused scale/offset (uint8_t)
  UInt8FusedQTy,
  // 8-bit quantized type with fused FP16 scale/offset (uint8_t)
  UInt8FusedFP16QTy,
  // 4-bit quantized type with fused FP16 scale/offset (uint8_t, each byte
  // represents 2 4-bit quantized data)
  UInt4FusedFP16QTy,
  // 4-bit quantized type with fused FP32 scale/offset (uint8_t, each byte
  // represents 2 4-bit quantized data)
  UInt4FusedQTy,
  // Bool type (bool)
  BoolTy,
};

/*@brief returns is \p elk is a quantized ElemKind.
 */
inline constexpr bool isQuantizedElemKind(dnn_lib::ElemKind elk) {
  if (elk == dnn_lib::ElemKind::Int8QTy || elk == dnn_lib::ElemKind::UInt8QTy || elk == dnn_lib::ElemKind::Int16QTy ||
      elk == dnn_lib::ElemKind::Int32QTy || elk == dnn_lib::ElemKind::UInt8FusedQTy ||
      elk == dnn_lib::ElemKind::UInt8FusedFP16QTy || elk == dnn_lib::ElemKind::UInt4FusedFP16QTy) {
    return true;
  } else {
    return false;
  }
}

/*@brief returns whether \p elk is an "index" ElemKind.
 */
inline constexpr bool isIndexElemKind(dnn_lib::ElemKind elk) {
  return elk == dnn_lib::ElemKind::Int32ITy or elk == dnn_lib::ElemKind::Int64ITy;
}

/*@brief returns wheter \p elk is a fused quantized ElemKind.
 */
inline constexpr bool isFusedQuantizedElemKind(ElemKind elk) {
  if (elk == dnn_lib::ElemKind::UInt8FusedQTy || elk == dnn_lib::ElemKind::UInt8FusedFP16QTy ||
      elk == dnn_lib::ElemKind::UInt4FusedFP16QTy) {
    return true;
  } else {
    return false;
  }
}
} // namespace dnn_lib

#endif // LIB_TYPES_H
