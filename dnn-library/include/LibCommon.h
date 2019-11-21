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

namespace dnn_lib {

inline __attribute__((always_inline)) void
fpReciprocalSingleElement(float val, float &recval) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "frcp.ps %[recval], %[val] \n"
                       : [ recval ] "=&f"(recval)
                       : [ val ] "f"(val));
}

inline __attribute__((always_inline)) void
fpPowSingleElement(float val1, float val2, float &res) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "flog.ps %[res], %[val1] \n"
                       "fmul.ps %[res], %[res], %[val2] \n"
                       "fexp.ps %[res], %[res] \n"
                       : [ res ] "=&f"(res)
                       : [ val1 ] "f"(val1), [ val2 ] "f"(val2));
}

inline __attribute__((always_inline))
void fpLog2SingleElement(float val, float &res) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "flog.ps %[res], %[val] \n"
                       : [ res ] "=&f"(res)
                       : [ val ] "f"(val));
}

inline __attribute__((always_inline))
void fpAddSingleElement(float val1, float val2, float &res) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fadd.ps %[res], %[val1], %[val2] \n"
                       : [ res ] "=&f"(res)
                       : [ val1 ] "f"(val1), [ val2 ] "f"(val2));
}

inline __attribute__((always_inline))
void loadFp32FromMemory(uint64_t addr, float &res) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "flw.ps %[res], 0(%[addr]) \n"
                       : [ res ] "=&f"(res)
                       : [ addr ] "r"(addr));
}

inline __attribute__((always_inline))
void convertFp16ToFp32(float src, float &dst) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.ps.f16 %[dst], %[src] \n"
                       : [ dst ] "=&f"(dst)
                       : [ src ] "f"(src));
}

inline __attribute__((always_inline))
void convertFp16ToFp32(uint16_t src, float &dst) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.ps.f16 %[dst], %[src] \n"
                       : [ dst ] "=&f"(dst)
                       : [ src ] "f"((float)src));
}

inline __attribute__((always_inline))
void convertFp32ToFp16(float src, float &dst) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.f16.ps %[dst], %[src] \n"
                       : [ dst ] "=&f"(dst)
                       : [ src ] "f"(src));
}
inline __attribute__((always_inline))
void convertFp32ToFp16(float src, uint16_t &dst) {
  float tmp;
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.f16.ps %[tmp], %[src] \n"
                       "fmv.x.w %[dst], %[tmp] \n"
                       : [ dst ] "=&r"(dst)
                       : [ src ] "f"(src), [ tmp ] "f"(tmp));
}

inline __attribute__((always_inline))
void storeFp32ToMemory(uint64_t addr, float val32) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fsw %[val32], 0(%[addr])\n"
                       :
                       : [ val32 ] "f"(val32), [ addr ] "r"(addr));
}

inline __attribute__((always_inline))
void storeFp16ToMemory(uint64_t addr, float val32) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       //"fsrli.pi %[val32], %[val32], 16 \n"
                       "fmvz.x.ps x1, %[val32], 0 \n"
                       "sh x1, 0(%[addr])\n"
                       :
                       : [ val32 ] "f"(val32), [ addr ] "r"(addr));
}

inline __attribute__((always_inline))
float loadAndConvertToFp32(uint64_t loadAddr) {
  float val16, val32;
  loadFp32FromMemory(loadAddr, val16);
  convertFp16ToFp32(val16, val32);
  return val32;
}

inline __attribute__((always_inline))
void storeFp32(uint64_t storeAddr, float val32) {
  storeFp32ToMemory(storeAddr, val32);
}

inline __attribute__((always_inline))
float loadAndConvertToFp16(uint64_t loadAddr) {
  float val16, val32;
  loadFp32FromMemory(loadAddr, val16);
  convertFp32ToFp16(val16, val32);
  return val32;
}

inline __attribute__((always_inline))
void storeFp16(uint64_t storeAddr, float val32) {
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

} // namespace dnn_lib

#endif /* LIB_COMMON_H */
