/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef WRITER_V2_H
#define WRITER_V2_H

#include "Float16.h"
#include "LibCommon.h"
#include "LibTypes.h"
#include <etsoc/isa/atomic.h>

namespace dnn_lib_v2 {

template <ElemKind elK, bool globalStore = false> class Writer {
private:
  using T = typename elemKind2elemTy<elK>::type;
  const float scale_;
  const int32_t offset_;
  T* const ptr_;

  inline __attribute__((always_inline)) static void write(T* ptr, T value) {
    constexpr size_t bytesPerElement = Type::getElementSize(elK);
    static_assert(bytesPerElement == 1 or bytesPerElement == 2 or bytesPerElement == 4 or bytesPerElement == 8);
    if constexpr (globalStore) {
      if constexpr (bytesPerElement == 1) {
        atomic_store_global_8(reinterpret_cast<uint8_t*>(ptr), static_cast<uint8_t>(value));
      } else if constexpr (bytesPerElement == 2) {
        atomic_store_global_16(reinterpret_cast<uint16_t*>(ptr), static_cast<uint16_t>(value));
      } else if constexpr (bytesPerElement == 4) {
        if constexpr (elK == FloatTy) {
          uint64_t x;
          __asm__("fmv.x.w %[destination], %[source]\n" : [ destination ] "=r"(x) : [ source ] "f"(value));
          atomic_store_global_32(reinterpret_cast<uint32_t*>(ptr), static_cast<uint32_t>(x));
        } else {
          atomic_store_global_32(reinterpret_cast<uint32_t*>(ptr), static_cast<uint32_t>(value));
        }
      } else if constexpr (bytesPerElement == 8) {
        static_assert(elK == Int64ITy);
        atomic_store_global_64(reinterpret_cast<uint64_t*>(ptr), static_cast<uint64_t>(value));
      }
    } else {
      *ptr = value;
    }
  }

public:
  Writer(T* ptr, float scale = 1.0, int32_t offset = 0)
    : scale_(scale)
    , offset_(offset)
    , ptr_(ptr) {
  }

#define WHEN(cond) template <ElemKind U = elK, typename std::enable_if<cond, size_t>::type = 0>

  WHEN(U == Float16Ty)
  inline __attribute__((always_inline)) Writer& operator=(float value) {
    uint16_t v;
    dnn_lib::convertFp32ToFp16(value, v);
    write(ptr_, v);
    return *this;
  }

  WHEN(isQuantizedElemKind(U))
  inline __attribute__((always_inline)) Writer& operator=(float value) {
    T quantizedValue = dnn_lib::quantize<T>(value, scale_, offset_);
    write(ptr_, quantizedValue);
    return *this;
  }

  WHEN(U != Float16Ty and not isQuantizedElemKind(U))
  inline __attribute__((always_inline)) Writer& operator=(T value) {
    write(ptr_, value);
    return *this;
  }

#undef WHEN
};

} // namespace dnn_lib_v2

#endif /* WRITER_V2_H */