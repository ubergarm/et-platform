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

#include "Compiler.h"
#include <array>
#include <stdint.h>

#define INLINE_ATTR __attribute__((always_inline)) inline

#define LOG2_CACHE_LINE_BYTES 6
#define CACHE_LINE_BYTES (1 << LOG2_CACHE_LINE_BYTES)

#define VLEN 256
#define VREG_BYTES (VLEN / 8)

#define MIN_PER_SHIRE 32

#ifndef NUM_HARTS
// FIXME, need to properly modify build-system to  include system/layout.h from et-common-libs
#define NUM_SHIRES 33
#define HARTS_PER_SHIRE 32 * 2
#define MIN_PER_SHIRE 32
#define NUM_HARTS (NUM_SHIRES * HARTS_PER_SHIRE)
#endif

//#define DIM_T_32
namespace dnn_lib {

#ifdef DIM_T_32
using dim_t = uint32_t;
using sdim_t = int32_t;
#else
using dim_t = size_t;
using sdim_t = int64_t;
#endif

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

template <ElemKind elK> struct elemKind2elemTy {
  using type = typename std::conditional<
    elK == ElemKind::FloatTy, float,
    typename std::conditional<
      elK == ElemKind::Float16Ty, uint16_t,
      typename std::conditional<
        elK == ElemKind::BFloat16Ty, uint16_t,
        typename std::conditional<
          elK == ElemKind::Int8QTy, int8_t,
          typename std::conditional<
            elK == ElemKind::UInt8QTy, uint8_t,
            typename std::conditional<
              elK == ElemKind::Int16QTy, int16_t,
              typename std::conditional<
                elK == ElemKind::Int32QTy, int32_t,
                typename std::conditional<
                  elK == ElemKind::Int32ITy, int32_t,
                  typename std::conditional<
                    elK == ElemKind::Int64ITy, int64_t,
                    typename std::conditional<
                      elK == ElemKind::UInt8FusedQTy, uint8_t,
                      typename std::conditional<
                        elK == ElemKind::UInt8FusedFP16QTy, uint8_t,
                        typename std::conditional<elK == ElemKind::UInt4FusedFP16QTy, uint8_t,
                                                  typename std::conditional<elK == ElemKind::BoolTy, bool,
                                                                            void // void is the default value, if no
                                                                                 // elKind matches
                                                                            >::type>::type>::type>::type>::type>::
                  type>::type>::type>::type>::type>::type>::type>::type;

  //@TODO static_assert(!std::is_same<type, void>::value);
};

// enum class PrecisionMode {
//  //TODO: Get same enumerate as Jitter
//  PM_FP_32   = 0,   // fp32
//  PM_FP_16   = 1,   // fp16
//  PM_BFP_16  = 2,   // bfloat16
//  PM_INT_32  = 3,   // quant int32
//  PM_INT_8   = 4,   // quant int8
//  PM_INT_16  = 5,   // quant int16
//  PM_INT_I32 = 6,   // idx int32
//  PM_INT_I64 = 7,   // idx int64
//  PM_UINT_8  = 8,   // quant uint8
//  PM_UINT_4  = 9,   // quant uint4
//  PM_BOOL    = 10,  // bool
//  MAX_PRECISION_MODES
// };

// template<class T>
// constexpr std::size_t getsize() {
//   return sizeof(T);
// }
// template<>
// constexpr std::size_t getsize<float16>() {
//   return 2;
// }

template <bool> struct conditional_;
template <> struct conditional_<false> { template <typename, typename T> using apply = T; };

template <bool V, typename T, typename F> using conditional_t = typename conditional_<V>::template apply<T, F>;

/*@brief returns whether \p elk is a quantized ElemKind.
 */
INLINE_ATTR constexpr bool isQuantizedElemKind(dnn_lib::ElemKind elk) {
  if (elk == dnn_lib::ElemKind::Int8QTy || elk == dnn_lib::ElemKind::UInt8QTy || elk == dnn_lib::ElemKind::Int16QTy ||
      elk == dnn_lib::ElemKind::Int32QTy || elk == dnn_lib::ElemKind::UInt8FusedQTy ||
      elk == dnn_lib::ElemKind::UInt8FusedFP16QTy || elk == dnn_lib::ElemKind::UInt4FusedFP16QTy)
    return true;
  else
    return false;
}

/*@brief returns whether \p elk is an "index" ElemKind.
 */
INLINE_ATTR constexpr bool isIndexElemKind(dnn_lib::ElemKind elk) {
  return elk == dnn_lib::ElemKind::Int32ITy or elk == dnn_lib::ElemKind::Int64ITy;
}

/*@brief returns wheter \p elk is a fused quantized ElemKind.
 */
INLINE_ATTR bool isFusedQuantizedElemKind(ElemKind elk) {
  if (elk == dnn_lib::ElemKind::UInt8FusedQTy || elk == dnn_lib::ElemKind::UInt8FusedFP16QTy ||
      elk == dnn_lib::ElemKind::UInt4FusedFP16QTy)
    return true;
  else
    return false;
}

template <size_t depth> struct IntTypeByDepth {};
template <> struct IntTypeByDepth<8> { using type = int8_t; };
template <> struct IntTypeByDepth<16> { using type = int16_t; };
template <> struct IntTypeByDepth<32> { using type = int32_t; };
template <> struct IntTypeByDepth<64> { using type = int64_t; };

template <ElemKind elemKindTy, ElemKind biasElemType> class AccumulatingQuantizedOpTypes {
private:
  static_assert((elemKindTy == ElemKind::Int8QTy and biasElemType == ElemKind::Int8QTy) or
                (elemKindTy == ElemKind::Int8QTy and biasElemType == ElemKind::Int32QTy) or
                (elemKindTy == ElemKind::Int16QTy and biasElemType == ElemKind::Int16QTy) or
                (elemKindTy == ElemKind::Int16QTy and biasElemType == ElemKind::Int32QTy));

  // \brief returns element (source or destination) bit depth
  static constexpr size_t getElemDepth() {
    size_t result = 8;
    if constexpr (elemKindTy == ElemKind::Int8QTy and biasElemType == ElemKind::Int8QTy) {
      result = 8;
    } else if constexpr (elemKindTy == ElemKind::Int8QTy and biasElemType == ElemKind::Int32QTy) {
      result = 8;
    } else if constexpr (elemKindTy == ElemKind::Int16QTy and biasElemType == ElemKind::Int16QTy) {
      result = 16;
    } else if constexpr (elemKindTy == ElemKind::Int16QTy and biasElemType == ElemKind::Int32QTy) {
      result = 16;
    }
    return result;
  }

  // \brief returns accumulator bit depth
  static constexpr size_t getAccumulatorDepth() {
    size_t result = 8;
    if constexpr (elemKindTy == ElemKind::Int8QTy and biasElemType == ElemKind::Int8QTy) {
      result = 32;
    } else if constexpr (elemKindTy == ElemKind::Int8QTy and biasElemType == ElemKind::Int32QTy) {
      result = 32;
    } else if constexpr (elemKindTy == ElemKind::Int16QTy and biasElemType == ElemKind::Int16QTy) {
      result = 64;
    } else if constexpr (elemKindTy == ElemKind::Int16QTy and biasElemType == ElemKind::Int32QTy) {
      result = 64;
    }
    return result;
  }

  // \brief returns bias bit depth
  static constexpr size_t getBiasDepth() {
    size_t result = 8;
    if constexpr (elemKindTy == ElemKind::Int8QTy and biasElemType == ElemKind::Int8QTy) {
      result = 8;
    } else if constexpr (elemKindTy == ElemKind::Int8QTy and biasElemType == ElemKind::Int32QTy) {
      result = 32;
    } else if constexpr (elemKindTy == ElemKind::Int16QTy and biasElemType == ElemKind::Int16QTy) {
      result = 16;
    } else if constexpr (elemKindTy == ElemKind::Int16QTy and biasElemType == ElemKind::Int32QTy) {
      result = 32;
    }
    return result;
  }

  static constexpr size_t elemDepth = getElemDepth();
  static constexpr size_t accumulatorDepth = getAccumulatorDepth();
  static constexpr size_t biasDepth = getBiasDepth();

public:
  using elemType = typename IntTypeByDepth<elemDepth>::type;
  using accumulatorType = typename IntTypeByDepth<accumulatorDepth>::type;
  using biasType = typename IntTypeByDepth<biasDepth>::type;
};

} // namespace dnn_lib

namespace dnn_lib_v2 {

typedef unsigned int u32_t;
typedef signed int s32_t;
typedef float f32_t;

enum class ETSoC1ElementKind {
  none,
  u32, // Unsigned 32 bits integer
  s32, // Singed 32 bits integer
  f32, // Single-precision or 32 bits floating point
};

template <ETSoC1ElementKind elk> struct s { using type = void; };
template <> struct s<ETSoC1ElementKind::u32> { using type = u32_t; };
template <> struct s<ETSoC1ElementKind::s32> { using type = s32_t; };
template <> struct s<ETSoC1ElementKind::f32> { using type = f32_t; };

#if COMPILER_CLANG
template <unsigned int n, typename T> struct v {
  static const int length = n;
  using type = T __attribute__((vector_size(sizeof(T) * n)));
};
#else
template <unsigned int n, typename T> struct v {
  static const int length = 1;
  using type = float;
};
#endif

using v8s32_t = v<8, s32_t>::type;
using v8u32_t = v<8, u32_t>::type;
using v8f32_t = v<8, f32_t>::type;

} // namespace dnn_lib_v2

namespace dnn_lib {
// Vector datatypes
using f32x8 = dnn_lib_v2::v8f32_t;
} // namespace dnn_lib

#endif // LIB_TYPES_H