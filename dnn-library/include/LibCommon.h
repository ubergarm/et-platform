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

#ifndef LIB_COMMON_H
#define LIB_COMMON_H

#include <cmath>
#include <limits>
#include <cstring>
#include <type_traits>
#include <algorithm>

#include "LibTypes.h"
#include "Float16.h"

#define INLINE_ATTR __attribute__((always_inline)) inline

namespace dnn_lib {

template <typename T, typename U>
inline T bitwise_copy(const U &x)
{
    static_assert(std::is_trivially_copyable<T>::value && std::is_trivially_copyable<U>::value, "pseudo_cast can't handle types which are not trivially copyable");
    static_assert(sizeof(T) == sizeof(U), "pseudo_cast can't handle types with different size");

    T to;
    memcpy(&to, &x, sizeof(T));
    return to;
}

template <typename T, typename U>
inline T bitwise_lsb_copy(const U &x)
{
    static_assert(std::is_trivially_copyable<T>::value && std::is_trivially_copyable<U>::value, "pseudo_cast can't handle types which are not trivially copyable");
    static_assert(sizeof(U) >= sizeof(T), "src type has to be equal or wider than dst type");

    T to;
    memcpy(&to, &x, sizeof(T));
    return to;
}


inline __attribute__((always_inline)) void
fpReciprocalSingleElement(float val, float &recval) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "frcp.ps %[recval], %[val] \n"
                       : [ recval ] "=f"(recval)
                       : [ val ] "f"(val));
}

inline __attribute__((always_inline)) void
fpPowSingleElement(float val1, float val2, float &res) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "flog.ps %[res], %[val1] \n"
                       "fmul.s %[res], %[res], %[val2] \n"
                       "fexp.ps %[res], %[res] \n"
                       : [ res ] "=&f"(res)
                       : [ val1 ] "f"(val1), [ val2 ] "f"(val2));
}

inline __attribute__((always_inline))
void fpLog2SingleElement(float val, float &res) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "flog.ps %[res], %[val] \n"
                       : [ res ] "=f"(res)
                       : [ val ] "f"(val));
}

inline __attribute__((always_inline))
void fpAddSingleElement(float val1, float val2, float &res) {
  __asm__ __volatile__("fadd.s %[res], %[val1], %[val2] \n"
                       : [ res ] "=f"(res)
                       : [ val1 ] "f"(val1), [ val2 ] "f"(val2));
}

inline __attribute__((always_inline))
void loadFp32FromMemory(float* addr, float &res) {
#if 0
  // code for simd (as with mask !=1)
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "flw.ps %[res], %[addr] \n"
                       : [ res ] "=f"(res)
                       : [addr] "m" (*(const float (*)[8]) addr)
                       );
#else
   res = *addr;
#endif

}

inline __attribute__((always_inline))
void convertFp16ToFp32(float src, float &dst) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.ps.f16 %[dst], %[src] \n"
                       : [ dst ] "=f"(dst)
                       : [ src ] "f"(src));
}

inline __attribute__((always_inline))
void convertFp16ToFp32(uint16_t src, float &dst) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fmv.s.x %[dst], %[src]\n"
                       "fcvt.ps.f16 %[dst], %[dst] \n"
                       : [ dst ] "=f"(dst)
                       : [src] "r" (src));
}

inline __attribute__((always_inline))
void convertFp32ToFp16(float src, float &dst) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.f16.ps %[dst], %[src] \n"
                       : [ dst ] "=f"(dst)
                       : [ src ] "f"(src));
}

inline __attribute__((always_inline))
void convertFp32ToFp16(float src, uint16_t &dst) {
  float tmp;
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.f16.ps %[tmp], %[src] \n"
                       "fmv.x.w %[dst], %[tmp] \n"
                       : [ dst ] "=r"(dst), [ tmp ] "=f" (tmp)
                       : [ src ] "f"(src));
}

inline __attribute__((always_inline))
void storeFp32ToMemory(float* addr, float val32) {
#if 0
  // code for simd (mask != 1)
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fsw %[val32], %[addr]\n"
                       : [addr] "=m" (*(float (*)[8]) addr)
                       : [ val32 ] "f"(val32));
#else
  *addr = val32;
#endif
                         
}

inline __attribute__((always_inline))
void storeFp16ToMemory(uint16_t *addr, float val32) {
#if 0
  // code for simd
  uint64_t tmp;
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fmvz.x.ps %[tmp], %[val32], 0 \n"
                       "sh %[tmp], %[addr]\n"
                       : [addr] "=m" (*(uint16_t (*)[8]) addr), [tmp] "+r" (tmp)
                       : [ val32 ] "f"(val32)
                       );
#else
  *addr = static_cast<uint16_t>(bitwise_copy<uint32_t>(val32));
#endif
}

inline __attribute__((always_inline))
float loadAndConvertToFp32(float* loadAddr) {
  float val16, val32;
  loadFp32FromMemory(loadAddr, val16);
  convertFp16ToFp32(val16, val32);
  return val32;
}

inline __attribute__((always_inline))
void storeFp32(float* storeAddr, float val32) {
  storeFp32ToMemory(storeAddr, val32);
}

inline __attribute__((always_inline))
float loadAndConvertToFp16(float* loadAddr) {
  float val16, val32;
  loadFp32FromMemory(loadAddr, val16);
  convertFp32ToFp16(val16, val32);
  return val32;
}

inline __attribute__((always_inline))
void storeFp16(uint16_t *storeAddr, float val32) {
  storeFp16ToMemory(storeAddr, val32);
}

inline __attribute__((always_inline))
void getReciprocal(float val, float &recval) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "frcp.ps %[recval], %[val] \n"
                       : [ recval ] "=f"(recval)
                       : [ val ] "f"(val));
}

/// \returns the value \p in as clipped to the range of \p DestTy.
template <class SrcTy, class DestTy>
inline __attribute__((always_inline))
DestTy clip(SrcTy in) {
  static_assert(sizeof(SrcTy) >= sizeof(DestTy), "Invalid types");
  auto mx = std::numeric_limits<DestTy>::max();
  auto mn = std::numeric_limits<DestTy>::min();
  return std::max<SrcTy>(mn, std::min<SrcTy>(mx, in));
}

/// Converts floating point value to DestTy (int8 or int32) based on the
/// quantization parameters \p TQP.
template <class DestTy>
inline __attribute__((always_inline))
DestTy quantize(float input, float scale, int32_t offset) {
  float invertedScale;
  fpReciprocalSingleElement(scale, invertedScale);
  float result = input * invertedScale + offset;
  return clip<int32_t, DestTy>((int32_t)nearbyintf(result));
}

// TODO Convert to int64_t
template <class SrcTy>
inline __attribute__((always_inline))
float dequantize(SrcTy input, float scale, int32_t offset) {
  return scale * ((int32_t)input - offset);
}

/// Converts a quantized value (type eTy) to floating point based on the
/// quantization parameters \p scale and \p offset. If the input type is int8_t,
/// then an offset of 128 is added to convert to uint8_t.
template <class eTy>
inline __attribute__((always_inline))
float dequantizeWithFloatOffset(eTy input, float scale,
                                                float offset) {
  uint8_t d = static_cast<uint8_t>(input);
  if (std::is_same<int8_t, eTy>::value) {
    d += 128;
  }
  return (d * scale) + offset;
}

inline __attribute__((always_inline))
int8_t quantizeValInt8(float val, float scale, int32_t offset) {
  return quantize<int8_t>(val, scale, offset);
}

using dim_t = uint64_t;
using sdim_t = int64_t;

/**
 * @brief Using C++ template for size-based type selection.
 *
 * Standardisation of size-specific integer types in C/C++ is extremely useful
 * for portability and avoiding Addresser class always that it is possible for
 * perfomance increase. It will depends on kind of node operation.
 * The workhorse of the code is the contional template.
 * "std::conditional<bool condition, typename ifTrue, typename ifFalse>"
 * 
 * @warning 
 *
 * @note One limitation to note is that this only goes up to 8 bytes(64 bits).
 *
 * @tparam T_numBytes  byte size to be choosen.
 */
template <std::uint8_t T_numBytes>
using UintSelector = 
  typename std::conditional<T_numBytes == 1, uint8_t,
    typename std::conditional<T_numBytes == 2, uint16_t,
      typename std::conditional<T_numBytes == 4, uint32_t, 
           std::uint64_t
      >::type
    >::type
  >::type;

} // namespace dnn_lib

#endif /* LIB_COMMON_H */
