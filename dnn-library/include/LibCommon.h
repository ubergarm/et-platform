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

#ifndef LIB_COMMON_H
#define LIB_COMMON_H

#include "Float16.h"
#include "LibTensor.h"
#include "LibTypes.h"
#include <algorithm>
#include <cmath>
#include <etsoc/isa/hart.h>
#include <limits>
#include <string.h>
#include <type_traits>

#include <etsoc/isa/tensors.h>
// Shall match wrappers in UberKernel.cc
extern void log_enter_user_region_wrapper(uint64_t ptr, uint16_t regionId);
extern void log_exit_user_region_wrapper(uint64_t ptr, uint16_t regionId);

namespace dnn_lib {

template <typename DestT, typename SrcT> inline __attribute__((always_inline)) DestT bitwise_copy(const SrcT& value) {

  static_assert(std::is_trivially_copyable<DestT>::value && std::is_trivially_copyable<SrcT>::value,
                "pseudo_cast can't handle types which are not trivially copyable");
  static_assert(sizeof(DestT) >= sizeof(SrcT), "Destination should be bigger or equally sized as source");

  DestT result;
  if constexpr (sizeof(DestT) > sizeof(SrcT)) {
    result = 0;
  }
  memcpy(&result, &value, sizeof(SrcT));

  return result;
}

template <bool setMask = true>
inline __attribute__((always_inline)) void fpReciprocalSingleElement(float val, float& recval) {
  if constexpr (setMask) {
    mask_set(0, 0x1);
  }
  __asm__ __volatile__("frcp.ps %[recval], %[val] \n" : [ recval ] "=f"(recval) : [ val ] "f"(val));
}

template <bool setMask = true>
inline __attribute__((always_inline)) void fpPowSingleElement(float val1, float val2, float& res) {
  if constexpr (setMask) {
    mask_set(0, 0x1);
  }
  __asm__ __volatile__("flog.ps %[res], %[val1] \n"
                       "fmul.s %[res], %[res], %[val2] \n"
                       "fexp.ps %[res], %[res] \n"
                       : [ res ] "=&f"(res)
                       : [ val1 ] "f"(val1), [ val2 ] "f"(val2));
}

template <bool setMask = true> inline __attribute__((always_inline)) void fpLog2SingleElement(float val, float& res) {
  if constexpr (setMask) {
    mask_set(0, 0x1);
  }
  __asm__ __volatile__("flog.ps %[res], %[val] \n" : [ res ] "=f"(res) : [ val ] "f"(val));
}

template <bool setMask = true>
inline __attribute__((always_inline)) void fpAddSingleElement(float val1, float val2, float& res) {
  if constexpr (setMask) {
    mask_set(0, 0x1);
  }
  __asm__ __volatile__("fadd.s %[res], %[val1], %[val2] \n"
                       : [ res ] "=f"(res)
                       : [ val1 ] "f"(val1), [ val2 ] "f"(val2));
}

inline __attribute__((always_inline))
void loadFp32FromMemory(float* addr, float &res) {
   res = *addr;
}

inline __attribute__((always_inline))
void storeFp32ToMemory(float* addr, float val32) {
  *addr = val32;
}

inline __attribute__((always_inline))
void storeFp16ToMemory(uint16_t *addr, float val32) {
  *addr = static_cast<uint16_t>(bitwise_copy<uint32_t>(val32));
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

// TODO: avoid redundant APIs
template <bool setMask = true> inline __attribute__((always_inline)) void getReciprocal(float val, float& recval) {
  fpReciprocalSingleElement<setMask>(val, recval);
}

/// \returns the value \p in as clipped to the range of \p DestTy.
template <class SrcTy, class DestTy>
inline __attribute__((always_inline))
DestTy clip(SrcTy in) {
  static_assert(sizeof(SrcTy) >= sizeof(DestTy), "Invalid types");
  auto mx = std::numeric_limits<DestTy>::max();
  auto mn = std::numeric_limits<DestTy>::min();
  return static_cast<DestTy>(std::max<SrcTy>(mn, std::min<SrcTy>(mx, in)));
}

/// Converts floating point value to DestTy (int8 or int32) based on the
/// quantization parameters \p TQP.
template <class DestTy>
inline __attribute__((always_inline))
DestTy quantize(float input, float scale, int32_t offset) {
  float invertedScale;
  fpReciprocalSingleElement(scale, invertedScale);
  float result = input * invertedScale + static_cast<float>(offset);
  return clip<int32_t, DestTy>((int32_t)nearbyintf/* round */(result));
}

// TODO Convert to int64_t
template <class SrcTy>
inline __attribute__((always_inline))
float dequantize(SrcTy input, float scale, int32_t offset) {
  return scale * static_cast<float>(static_cast<int32_t>(input) - offset);
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

/*@brief newDims filled with currDims values and expanded dimensions =1
 *until max tensor dimension is reached.
 */
inline __attribute__((always_inline)) void expandDimsToMax(dim_t* newDims, dim_t* currDims, dim_t numDims) {

  for (dim_t i = 0; i < max_tensor_dimensions; i++) {
    if (i< numDims)
      newDims[i] = currDims[i];
    else
      newDims[i] = 1;
  }
}

/*@brief The axis value set which dimension has to be twisted
 *It works up to max_tensor_dimensions=6
 */
template <typename ElemTy>
inline __attribute__((always_inline)) void loopAxis(Handle<ElemTy> srcH, Handle<ElemTy> destH,
                                                    const dim_array_t& newDims, dim_t axis) {

  dim_array_t indicesDest = {0,};
  dim_array_t indicesSrc = {0,};
  dim_t ndx[max_tensor_dimensions] = {0,};
  
  
  for (ndx[0] = 0; ndx[0] < newDims[0]; ndx[0]++)
    for (ndx[1] = 0; ndx[1] < newDims[1]; ndx[1]++)
      for (ndx[2] = 0; ndx[2] < newDims[2]; ndx[2]++)
        for (ndx[3] = 0; ndx[3] < newDims[3]; ndx[3]++)
          for (ndx[4] = 0; ndx[4] < newDims[4]; ndx[4]++)
            for (ndx[5] = 0; ndx[5] < newDims[5]; ndx[5]++) {

              for (uint8_t i = 0; i < max_tensor_dimensions; i++) {
                indicesSrc.data()[i] = ndx[i];
                if ( i != axis)
                  indicesDest[i] = ndx[i];
                else                  
                  indicesDest[i] = newDims[i]-1-ndx[i];
              }
              /* auto x = srcH.getIterator(indicesSrc); */
        /* auto y = destH.getIterator(indicesDest); */
        //*y = *x;
        
              destH.at(indicesDest) = srcH.at(indicesSrc);
            }
}

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

// TODO: [SW-13991]
// nullptr global initialized pointers are emitted to bss, and as
// of today, bss is not 0-initialized on etsoc, so program semantics
// is not preserved, we can't invalidate pointers here with nullptr
static void* invalidPtr = (void*)0x1;

/* 1 pointer enty per cache-line*/
struct alignas(CACHE_LINE_BYTES) HartLog {
  void* ptr_ = invalidPtr;
};

#define DNNLIB_PROFILING
#ifdef DNNLIB_PROFILING

/**
 * per-hart array singleton of log pointers
 */
[[maybe_unused]] alignas(CACHE_LINE_BYTES) extern HartLog hartLogs[NUM_HARTS];

/**
 * @brief dnnlib adapter to  log_enter_user_region
 * @param regionId: numeric id of the regiion.
 */
INLINE_ATTR void logEnterUserRegion(uint16_t regionId) {
  auto logPtr = hartLogs[get_hart_id()].ptr_;
  if (logPtr != invalidPtr) {
    ::log_enter_user_region_wrapper(uint64_t(logPtr), regionId);
  }
}

/**
 * @brief dnnlib adapter to  log_exit_user_region
 * @param regionId: numeric id of the region.
 */
INLINE_ATTR void logExitUserRegion(uint16_t regionId) {
  auto logPtr = hartLogs[get_hart_id()].ptr_;
  if (logPtr != invalidPtr) {
    ::log_exit_user_region_wrapper(uint64_t(logPtr), regionId);
  }
}

#else
INLINE_ATTR void logEnterUserRegion([[maybe_unused]] uint16_t regionId) {
}
INLINE_ATTR void logExitUserRegion([[maybe_unused]] uint16_t regionId) {
}
#endif

} // namespace dnn_lib

#endif /* LIB_COMMON_H */
