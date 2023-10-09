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

#ifndef FLOAT16_H
#define FLOAT16_H

#include "Compiler.h"
#include <cstdint>
#include <etsoc/isa/tensors.h>

static const float MAX_FP16_DENORM = ((float(1 << 10) - 1) / float(1 << 24)); // maximum fp16 denormal = 2^-14 - 2^-24

namespace dnn_lib {

template <bool setMask = true> inline __attribute__((always_inline)) void convertFp16ToFp32(float src, float& dst) {
#if COMPILER_GCC
  if constexpr (setMask) {
    mask_set(0, 0x1);
  }
#endif
  __asm__ __volatile__("fcvt.ps.f16 %[dst], %[src] \n"
                       : [ dst ] "=f"(dst)
                       : FOR_CLANG([ vmask ] "M"(1)) FOR_CLANG_COMMA[src] "f"(src)
                       :);
}

template <bool setMask = true> inline __attribute__((always_inline)) void convertFp16ToFp32(uint16_t src, float& dst) {
#if COMPILER_GCC
  if constexpr (setMask) {
    mask_set(0, 0x1);
  }
#endif
  __asm__ __volatile__("fmv.s.x %[dst], %[src]\n"
                       "fcvt.ps.f16 %[dst], %[dst] \n"
                       : [ dst ] "=f"(dst)
                       : FOR_CLANG([ vmask ] "M"(1)) FOR_CLANG_COMMA[src] "r"(src)
                       :);
}

template <bool setMask = true> inline __attribute__((always_inline)) void convertFp32ToFp16(float src, float& dst) {
#if COMPILER_GCC
  if constexpr (setMask) {
    mask_set(0, 0x1);
  }
#endif
  __asm__ __volatile__("fcvt.f16.ps %[dst], %[src] \n"
                       : [ dst ] "=f"(dst)
                       : FOR_CLANG([ vmask ] "M"(1)) FOR_CLANG_COMMA[src] "f"(src)
                       :);
}

template <bool setMask = true> inline __attribute__((always_inline)) void convertFp32ToFp16(float src, uint16_t& dst) {
#if COMPILER_GCC
  if constexpr (setMask) {
    mask_set(0, 0x1);
  }
#endif
  float tmp;
  __asm__ __volatile__("fcvt.f16.ps %[tmp], %[src] \n"
                       "fmv.x.w %[dst], %[tmp] \n"
                       : [ dst ] "=r"(dst), [ tmp ] "=f"(tmp)
                       : FOR_CLANG([ vmask ] "M"(1)) FOR_CLANG_COMMA[src] "f"(src)
                       :);
}

template <bool setMask = true> inline __attribute__((always_inline)) void convertBfloat16ToFp32(float src, float& dst) {
#if COMPILER_GCC
  if constexpr (setMask) {
    mask_set(0, 0x1);
  }
#endif
  __asm__ __volatile__("fslli.pi %[dst], %[src], 16 \n"
                       : [ dst ] "=f"(dst)
                       : FOR_CLANG([ vmask ] "M"(1)) FOR_CLANG_COMMA[src] "f"(src)
                       :);
}

template <bool setMask = true>
inline __attribute__((always_inline)) void convertBfloat16ToFp32(uint16_t src, float& dst) {
#if COMPILER_GCC
  if constexpr (setMask) {
    mask_set(0, 0x1);
  }
#endif
  __asm__ __volatile__("fmv.s.x %[dst], %[src]\n"
                       "fslli.pi %[dst], %[dst], 16 \n"
                       : [ dst ] "=f"(dst)
                       : FOR_CLANG([ vmask ] "M"(1)) FOR_CLANG_COMMA[src] "r"(src)
                       :);
}

/// Use a proxy type in case we need to change it in the future.

using Float16Storage = float;

class float16 {
public:
  union {
    uint32_t uint32;
    float fp;
  } data_;

  inline __attribute__((always_inline)) float16(uint16_t data) {
    data_.uint32 = static_cast<uint32_t>(data);
  }

  inline __attribute__((always_inline)) float16(float data) {
    data_.fp = data;
  }

  inline __attribute__((always_inline)) float16() {
    data_.fp = 0.0;
  }

  inline __attribute__((always_inline)) float16& operator=(const float16& val) {
    data_ = val.data_;
    return *this;
  }

  inline __attribute__((always_inline)) float16& operator=(float val) {
    data_.fp = val;
    return *this;
  }

  /// Comparisons.
  inline __attribute__((always_inline)) bool operator<(const float16& b) const {
    return this->data_.fp < b.data_.fp;
  }
  inline __attribute__((always_inline)) bool operator>(const float16& b) const {
    return this->data_.fp > b.data_.fp;
  }
  inline __attribute__((always_inline)) bool operator==(const float16& b) const {
    return this->data_.fp == b.data_.fp;
  }
  inline __attribute__((always_inline)) bool operator<=(const float16& b) const {
    return this->data_.fp <= b.data_.fp;
  }
  inline __attribute__((always_inline)) bool operator>=(const float16& b) const {
    return this->data_.fp >= b.data_.fp;
  }

  /// Cast operators.

  inline __attribute__((always_inline)) float16 fp32_to_fp16_value() const {
    float16 ret;
    dnn_lib::convertFp32ToFp16(data_.fp, ret.data_.fp);
    return ret;
  }

  inline __attribute__((always_inline)) float fp16_to_fp32_value() const {
    float dst = 0.0;
    dnn_lib::convertFp16ToFp32(data_.fp, dst);
    return dst;
  }

  inline __attribute__((always_inline)) float convertFp32ToFp16() const {
    float dst;
    dnn_lib::convertFp32ToFp16(data_.fp, dst);
    return dst;
  }
};

class bfloat16 {};

} // namespace dnn_lib

#endif // FLOAT16_H