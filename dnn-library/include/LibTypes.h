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

#include <stdint.h>
#include <array>

#define INLINE_ATTR __attribute__((always_inline)) inline

#define CACHE_LINE_BYTES 64
#define MIN_PER_SHIRE 32

//#define DIM_T_32
namespace dnn_lib {

#ifdef DIM_T_32
using dim_t = uint32_t;
using sdim_t = int32_t;
#else
  using dim_t = size_t;
using sdim_t = int64_t;
#endif
 

  //@WARNING this enumerate is not the same as jitter defines ....!!
  //
/// It has to be an identically copy as it is in   glow_fork/include/glow/Base/Type.h
/// An enum representing the type used by the elements of a tensor. The types of
/// Handles for these tensors should match the element kind.
enum ElemKind : unsigned char {
  // 32-bit float type (float)
  FloatTy,
  // 16-bit float type (half, fp16)
  Float16Ty,
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
  // Bool type (bool)
  BoolTy,
};

// enum class PrecisionMode {
//   // TODO: Get same enumerate as Jitter
//   PM_FP_32 = 0,   // fp32
//   PM_FP_16 = 1,   // fp16
//   PM_INT_32 = 2,  // quant int32
//   PM_INT_8 = 3,   // quant int8
//   PM_INT_16 = 4,  // quant int16
//   PM_INT_I32 = 5, // idx int32
//   PM_INT_I64 = 6, // idx int64
//   PM_UINT_8 = 7,  // quant uint8
//   PM_BOOL = 8,    // bool
//   MAX_PRECISION_MODES
// };


// template<class T>
// constexpr std::size_t getsize() {
//   return sizeof(T);
// }
// template<>
// constexpr std::size_t getsize<float16>() {
//   return 2;
// }

  template <bool>
  struct conditional_;
  template<>
  struct conditional_<false> {
    template <typename, typename T>
    using apply = T;
  };

  template <bool V, typename T, typename F>
  using conditional_t = typename conditional_<V>::template apply<T, F>;
  
  /*@brief returns is \p elk is a quantized ElemKind.
   */
  inline bool isQuantizedElemKind(dnn_lib::ElemKind elk) {
    return (elk == dnn_lib::ElemKind::Int8QTy ||
            elk == dnn_lib::ElemKind::UInt8QTy ||
            elk == dnn_lib::ElemKind::Int16QTy ||
            elk == dnn_lib::ElemKind::Int32QTy ||
            elk == dnn_lib::ElemKind::UInt8FusedQTy ||
            elk == dnn_lib::ElemKind::UInt8FusedFP16QTy ||
            elk == dnn_lib::ElemKind::UInt4FusedFP16QTy);
  }

  /*@brief returns wheter \p elk is a fused quantized ElemKind.
   */
  inline bool isFusedQuantizedElemKind(ElemKind elk) {
    return (elk == dnn_lib::ElemKind::UInt8FusedQTy ||
            elk == dnn_lib::ElemKind::UInt8FusedFP16QTy ||
            elk == dnn_lib::ElemKind::UInt4FusedFP16QTy);
  }


  
}
#endif //LIB_TYPES_H
