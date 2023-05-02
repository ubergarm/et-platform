/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies adn
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _FFT_V2_H_
#define _FFT_V2_H_

#ifndef CACHE_LINE_BYTES
#define CACHE_LINE_BYTES 64
#endif

#include "FFTTables.h"
#include "LoadStore.h"
#include "utils.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <etsoc/isa/atomic.h>
#include <etsoc/isa/cacheops-umode.h>

#ifdef GPSDK
#include "sync.h"
#endif

#ifndef INLINE_ATTR
#define INLINE_ATTR __attribute__((always_inline)) inline
#endif

#ifdef FFT_HOST_TEST
#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)
#endif

namespace {

constexpr size_t kImageDefaultFFTSize = 256;

}

namespace dnn_lib_v2 {

class Stack {
private:
  using EType = uint32_t;
  static constexpr size_t elementsPerMinion = 16 * 1024;
  static constexpr size_t numMinions = 1024;
  static constexpr size_t elementsPerCacheLine = CACHE_LINE_BYTES / sizeof(EType);
  static_assert(CACHE_LINE_BYTES % sizeof(EType) == 0);

public:
  Stack(size_t minionId) {
    constexpr size_t totalElements = elementsPerMinion * numMinions;
    static EType allocation[totalElements] __attribute__((aligned(64)));
    pointer = allocation + elementsPerMinion * minionId;
    start = pointer;
  }

  ~Stack() {
    assert(pointer == start);
  }

  EType* current() const {
    return pointer;
  }

  void restore(EType* value) {
    pointer = value;
  }

  template <typename T, size_t elements = 1> T* push() {
    T* result = reinterpret_cast<T*>(pointer);
    constexpr size_t cacheLines = (sizeof(T) * elements + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
    pointer += cacheLines * elementsPerCacheLine;
    return result;
  }

  template <typename T> T* push(size_t elements) {
    T* result = reinterpret_cast<T*>(pointer);
    size_t cacheLines = (sizeof(T) * elements + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
    pointer += cacheLines * elementsPerCacheLine;
    assert(pointer <= start + elementsPerMinion);
    return result;
  }

  template <typename T, size_t elements = 1> void pop() {
    pointer -= sizeof(T) * elements;
  }

  template <typename T> static T* offset(T* value, size_t idOffset) {
    return value + idOffset * elementsPerMinion;
  }

private:
  EType* pointer;
  EType* start;
};

constexpr size_t countBits(size_t value) {
  size_t result = 0;
  while (value) {
    result += (value & 1);
    value >>= 1;
  }
  return result;
}

static_assert(countBits(0) == 0);
static_assert(countBits(1) == 1);
static_assert(countBits(8) == 1);
static_assert(countBits(9) == 2);

constexpr bool isPowerOfTwo(size_t value) {
  return countBits(value) == 1;
}

static_assert(isPowerOfTwo(0) == false);
static_assert(isPowerOfTwo(8) == true);
static_assert(isPowerOfTwo(9) == false);

constexpr size_t log2(size_t value) {
  size_t result = 0;
  while (value >>= 1) {
    result++;
  }
  return result;
}

static_assert(log2(1) == 0);
static_assert(log2(2) == 1);
static_assert(log2(4) == 2);

constexpr size_t countTrailingZeros(size_t value, size_t maxDigits) {
  size_t result = 0;
  size_t bit = 1;
  while ((value & bit) == 0 and result < maxDigits) {
    result++;
    bit <<= 1;
  }
  return result;
}

static_assert(countTrailingZeros(1, 2) == 0);
static_assert(countTrailingZeros(2, 0) == 0);
static_assert(countTrailingZeros(2, 2) == 1);
static_assert(countTrailingZeros(0, 2) == 2);
static_assert(countTrailingZeros(6, 3) == 1);

//  Compute 1.f / static_cast<float>(n) when n is a power of two without using divides
INLINE_ATTR float rec(size_t n) {
  assert(isPowerOfTwo(n));
  float result = 2.f;
  while (n) {
    result *= 0.5f;
    n >>= 1;
  }
  return result;
}

INLINE_ATTR void eulerFormula(float angle, float& real, float& img) {
  real = cos(angle);
  img = sin(angle);
}

INLINE_ATTR void w(size_t j, float recN, float& real, float& img) {
  return eulerFormula(-2.f * static_cast<float>(M_PI) * static_cast<float>(j) * recN, real, img);
}

INLINE_ATTR void twiddleVectorBig(size_t n, float real[], float img[]) {

  assert(n >= 16 and isPowerOfTwo(n));

  real[0] = 1;
  img[0] = 0;
  real[n >> 2] = 0;
  img[n >> 2] = -1;
  real[n >> 1] = -1;
  img[n >> 1] = 0;

  const float k = 2.f * static_cast<float>(M_PI) * rec(n);
  for (uint32_t j = 1; j < (n >> 3); ++j) {
    const float angle = k * static_cast<float>(j);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    real[n - j] = cosine;
    img[n - j] = sine;
    real[j] = cosine;
    img[j] = -sine;
    real[(n >> 2) - j] = sine;
    img[(n >> 2) - j] = -cosine;
    real[(n >> 2) + j] = -sine;
    img[(n >> 2) + j] = -cosine;
    real[(n >> 1) - j] = -cosine;
    img[(n >> 1) - j] = -sine;
    real[(n >> 1) + j] = -cosine;
    img[(n >> 1) + j] = sine;
    real[n - (n >> 2) - j] = -sine;
    img[n - (n >> 2) - j] = cosine;
    real[n - (n >> 2) + j] = sine;
    img[n - (n >> 2) + j] = cosine;
  }
  const float sine = sin(static_cast<float>(M_PI) * 0.25f);
  real[n >> 3] = sine;
  img[n >> 3] = -sine;
  real[(n >> 3) + (n >> 2)] = -sine;
  img[(n >> 3) + (n >> 2)] = -sine;
  real[(n >> 3) + (n >> 1)] = -sine;
  img[(n >> 3) + (n >> 1)] = sine;
  real[n - (n >> 3)] = sine;
  img[n - (n >> 3)] = sine;
}

INLINE_ATTR void twiddleVector(Stack& stack, size_t n, float*& real, float*& img) {

  assert(isPowerOfTwo(n));

  if (likely(n == kImageDefaultFFTSize)) {
    real = const_cast<float*>(tReal);
    img = const_cast<float*>(tImg);
  } else if (n > kImageDefaultFFTSize) {
    real = stack.push<float>(n);
    img = stack.push<float>(n);
    twiddleVectorBig(n, real, img);
  } else { // (n < kImageDefaultFFTSize)
    size_t n2nbits2 = log2(n);
    real = stack.push<float>(n);
    img = stack.push<float>(n);
    for (size_t i = 0; i < n; i++) {
      real[i] = (tReal[(kImageDefaultFFTSize >> n2nbits2) * i]);
      img[i] = (tImg[(kImageDefaultFFTSize >> n2nbits2) * i]);
    }
  }
}

INLINE_ATTR void mult(float x, float y, float u, float v, float& real, float& img) {
  real = x * u - y * v;
  img = x * v + y * u;
}

INLINE_ATTR void add(float x, float y, float u, float v, float& real, float& img) {
  real = x + u;
  img = y + v;
}

INLINE_ATTR void sub(float x, float y, float u, float v, float& real, float& img) {
  real = x - u;
  img = y - v;
}

constexpr size_t twiddleIndex(size_t round, size_t i) {
  return i & ((8 + 16 + 32 + 64 + 128) >> round); // TODO: make this function to work for sizes 512 and beyond!
}

INLINE_ATTR void fft16Round(const float* twiddleReal, const float* twiddleImg, float XReal[16], float XImg[16],
                            int32_t round, float resultReal[16], float resultImg[16], const int32_t selectMultSecond[8],
                            const int32_t selectAddOrSubFirst[8]) {
  // This loop is suitable for vectorizing
  for (size_t i = 0; i < 8; ++i) {
    size_t tidx = twiddleIndex(round, i);
    size_t sidx = selectMultSecond[i];
    size_t fidx = selectAddOrSubFirst[i];
    float termReal, termImg;
    mult(twiddleReal[tidx], twiddleImg[tidx], XReal[sidx], XImg[sidx], termReal, termImg);
    add(XReal[fidx], XImg[fidx], termReal, termImg, resultReal[i], resultImg[i]);
    sub(XReal[fidx], XImg[fidx], termReal, termImg, resultReal[i + 8], resultImg[i + 8]);
  }
}

INLINE_ATTR void fft16Slice(float* real, float* img, int32_t start, int32_t step, [[maybe_unused]] size_t size,
                            const float* twiddleReal, const float* twiddleImg, float resReal[16], float resImg[16]) {

  assert(size == 16);
  float tmpReal[16];
  float tmpImg[16];

  int32_t selectMultSecond[8] = {8, 12, 10, 14, 9, 13, 11, 15};
  int32_t selectAddOrSubFirst[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  // This loop is suitable for vectorizing
  for (size_t i = 0; i < 8; ++i) {
    selectMultSecond[i] = start + step * selectMultSecond[i];
    selectAddOrSubFirst[i] = start + step * selectAddOrSubFirst[i];
  }

  fft16Round(twiddleReal, twiddleImg, real, img, 0, tmpReal, tmpImg, selectMultSecond, selectAddOrSubFirst);

  constexpr int32_t selectMultSecond2[8] = {1, 3, 5, 7, 9, 11, 13, 15};
  constexpr int32_t selectAddOrSubFirst2[8] = {0, 2, 4, 6, 8, 10, 12, 14};

  fft16Round(twiddleReal, twiddleImg, tmpReal, tmpImg, 1, resReal, resImg, selectMultSecond2, selectAddOrSubFirst2);

  fft16Round(twiddleReal, twiddleImg, resReal, resImg, 2, tmpReal, tmpImg, selectMultSecond2, selectAddOrSubFirst2);

  fft16Round(twiddleReal, twiddleImg, tmpReal, tmpImg, 3, resReal, resImg, selectMultSecond2, selectAddOrSubFirst2);
}

INLINE_ATTR void reduce(const float* baseTwiddleReal, const float* baseTwiddleImg, size_t twiddleStep, size_t halfSize,
                        float* tmpRealEven, float* tmpImgEven, float* tmpRealOdd, float* tmpImgOdd, float* resultReal,
                        float* resultImg) {
  size_t twiddleIndex = 0;
  for (size_t j = 0; j < halfSize; ++j) {
    float twiddleReal = baseTwiddleReal[twiddleIndex];
    float twiddleImg = baseTwiddleImg[twiddleIndex];
    float termReal, termImg;
    mult(twiddleReal, twiddleImg, tmpRealOdd[j], tmpImgOdd[j], termReal, termImg);
    add(tmpRealEven[j], tmpImgEven[j], termReal, termImg, resultReal[j], resultImg[j]);
    sub(tmpRealEven[j], tmpImgEven[j], termReal, termImg, resultReal[j + (halfSize)], resultImg[j + (halfSize)]);
    twiddleIndex += twiddleStep;
  }
}

INLINE_ATTR void vectorReduce(const float* baseTwiddleReal, const float* baseTwiddleImg, size_t twiddleStep,
                              size_t halfSize, float* tmpRealEven, float* tmpImgEven, float* tmpRealOdd,
                              float* tmpImgOdd, float* resultReal, float* resultImg, const std::array<int32_t, 8>& vI) {

  constexpr int32_t simd_width = 8;
  const int32_t twiddleStride = static_cast<int32_t>(twiddleStep * sizeof(float) * simd_width);

  float* rEven = tmpRealEven;
  float* iEven = tmpImgEven;
  float* rOdd = tmpRealOdd;
  float* iOdd = tmpImgOdd;
  float* rResult = resultReal;
  float* iResult = resultImg;

  float* rResult2 = resultReal + halfSize;
  float* iResult2 = resultImg + halfSize;

  f32x8 tmp0, tmp1, tmp2, tmp3, tmp4;
  f32x8 twiddleIndexArray;

  // Init twiddleIndex
  __asm__ __volatile__(
    "flw.ps     %[twiddleIndex], %[vI]\n"                    // tmp0 <- load(*mulSecond);
    "fbcx.ps    %[tmp0], %[twiddleStep]\n"                   // tmp1 <- twiddleStep;
    "fmul.pi    %[twiddleIndex], %[twiddleIndex], %[tmp0]\n" // tmp0'twiddleIndex' <- vmul(twiddleStep, i);
    "fslli.pi   %[twiddleIndex], %[twiddleIndex], 2\n"
    : [ twiddleIndex ] "=&f"(twiddleIndexArray), [ tmp0 ] "=&f"(tmp0)
    : [ twiddleStep ] "r"(twiddleStep), [ vI ] "m"(vI));

  size_t j = 0;
  for (j = 0; j < halfSize - (simd_width - 1); j += simd_width) {
    f32x8 termReal, termImg;
    f32x8 twiddleReal, twiddleImg;
    f32x8 tmpAddReal, tmpAddImg, tmpSubReal, tmpSubImg;

    __asm__ __volatile__(
      "fgw.ps     %[twiddleReal], %[twiddleIndex](%[baseTwiddleReal])\n" // twiddleReal <- gather(@baseTwiddleReal,
                                                                         // twiddleIndex);
      "fgw.ps     %[twiddleImg], %[twiddleIndex](%[baseTwiddleImg])\n"   // twiddleImg <- gather(@baseTwiddleImg,
                                                                         // twiddleIndex);

      "flw.ps     %[tmp3], 0(%[tmpRealOdd])\n" // tmp3 <- load(*tmpRealOdd);
      "flw.ps     %[tmp4], 0(%[tmpImgOdd])\n"  // tmp4 <- load(*tmpImgOdd);

      // MULT: termReal
      "fmul.ps    %[termReal], %[twiddleImg], %[tmp4]\n"               // (twiddleImg * tmpImgOdd)
      "fmsub.ps   %[termReal], %[twiddleReal], %[tmp3], %[termReal]\n" // termReal <- twiddleReal * tmpRealOdd -
                                                                       // (twiddleImg * tmpImgOdd);
      // MULT: termImg
      "fmul.ps    %[termImg], %[twiddleReal], %[tmp4]\n"            // (twiddleReal * tmpImgOdd)
      "fmadd.ps   %[termImg], %[twiddleImg], %[tmp3], %[termImg]\n" // termImg <- twiddleImg * tmpImgOdd + (twiddleImg *
                                                                    // tmpRealOdd);

      // ADD
      "flw.ps     %[tmp1], 0(%[tmpRealEven])\n" // tmp1 <- load(*tmpRealEven);
      "flw.ps     %[tmp2], 0(%[tmpImgEven])\n"  // tmp2 <- load(*tmpImgEven);

      "fadd.ps    %[tmpAddReal], %[tmp1], %[termReal]\n" // resultReal[j] <- tmpRealEven + termReal;
      "fadd.ps    %[tmpAddImg], %[tmp2], %[termImg]\n"
      "fsw.ps     %[tmpAddReal], 0(%[resultReal])\n"
      "fsw.ps     %[tmpAddImg], 0(%[resultImg])\n"

      // SUB
      "fsub.ps    %[tmpSubReal], %[tmp1], %[termReal]\n" // resultReal[j+halfsize] <- tmpRealEven - termReal;
      "fsub.ps    %[tmpSubImg], %[tmp2], %[termImg]\n"
      "fsw.ps     %[tmpSubReal], 0(%[resultReal2])\n"
      "fsw.ps     %[tmpSubImg], 0(%[resultImg2])\n"

      // Stride twiddleIndex
      "fbcx.ps     %[tmp0], %[twiddleStride]\n"
      "fadd.pi     %[twiddleIndex], %[twiddleIndex], %[tmp0]\n"

      : [ twiddleIndex ] "+&f"(twiddleIndexArray), [ tmp0 ] "=&f"(tmp0), [ tmp1 ] "=&f"(tmp1), [ tmp2 ] "=&f"(tmp2),
        [ tmp3 ] "=&f"(tmp3), [ tmp4 ] "=&f"(tmp4), [ twiddleReal ] "=&f"(twiddleReal),
        [ twiddleImg ] "=&f"(twiddleImg), [ tmpAddReal ] "=&f"(tmpAddReal), [ tmpAddImg ] "=&f"(tmpAddImg),
        [ tmpSubReal ] "=&f"(tmpSubReal), [ tmpSubImg ] "=&f"(tmpSubImg), [ termReal ] "=&f"(termReal),
        [ termImg ] "=&f"(termImg)
      : [ twiddleStride ] "r"(twiddleStride), [ baseTwiddleReal ] "r"(baseTwiddleReal),
        [ baseTwiddleImg ] "r"(baseTwiddleImg), [ tmpRealOdd ] "r"(rOdd), [ tmpImgOdd ] "r"(iOdd),
        [ tmpRealEven ] "r"(rEven), [ tmpImgEven ] "r"(iEven), [ resultReal ] "r"(rResult), [ resultImg ] "r"(iResult),
        [ resultReal2 ] "r"(rResult2), [ resultImg2 ] "r"(iResult2)
      : "memory");

    // Increment pointers
    rEven += simd_width;
    iEven += simd_width;
    rOdd += simd_width;
    iOdd += simd_width;
    rResult += simd_width;
    iResult += simd_width;
    rResult2 += simd_width;
    iResult2 += simd_width;
  }

  // Loop unroll epilogue
  size_t twiddleIndexEpilogue = j * twiddleStep;
  for (; j < halfSize; j++) {
    float twiddleReal = baseTwiddleReal[twiddleIndexEpilogue];
    float twiddleImg = baseTwiddleImg[twiddleIndexEpilogue];
    float termReal, termImg;
    mult(twiddleReal, twiddleImg, tmpRealOdd[j], tmpImgOdd[j], termReal, termImg);
    add(tmpRealEven[j], tmpImgEven[j], termReal, termImg, resultReal[j], resultImg[j]);
    sub(tmpRealEven[j], tmpImgEven[j], termReal, termImg, resultReal[j + (halfSize)], resultImg[j + (halfSize)]);
    twiddleIndexEpilogue += twiddleStep;
  }
}

#ifndef FFT_HOST_TEST

INLINE_ATTR void vectorFft16Round(const float* twiddleReal, const float* twiddleImg, float XReal[16], float XImg[16],
                                  int32_t round, float resultReal[16], float resultImg[16],
                                  const int32_t selectMultSecond[8], const int32_t selectAddOrSubFirst[8]) {

  constexpr int32_t mask = 8 + 16 + 32 + 64 + 128;

  int32_t vI[8] = {0, 1, 2, 3, 4, 5, 6, 7};

  float twIndices;
  float twReal, twImg, xRealMul, xImgMul, xRealAdd, xImgAdd;
  float termReal, termImg;
  float tmp0, tmp1;
  float mulIndices, addSubIndices;

  __asm__ __volatile__(
    // compute twiddle gather indices
    "fbcx.ps    %[twIndices], %[mask]\n"               // twIndices <- 248;
    "fbcx.ps    %[tmp0], %[round]\n"                   //
    "fsra.pi    %[twIndices], %[twIndices], %[tmp0]\n" // twIndices <- twIndices >> round;
    "flw.ps     %[tmp1], 0(%[vI])\n"
    "fand.pi    %[twIndices], %[twIndices], %[tmp1]\n" // twIndices[i] <- twIndices[i] * i;
    // load multiply, add, sub gather indices
    "flw.ps     %[mulIndices], 0(%[selectMultSecond])\n"       // mulIndices <- load(@selectMultSecond, 8);
    "flw.ps     %[addSubIndices], 0(%[selectAddOrSubFirst])\n" // addSubIndices <- load(@selectAddOrSubFirst, 8);
    // gather twiddle values
    "fslli.pi   %[twIndices], %[twIndices], 2\n"           // twIndices << 2
    "fgw.ps     %[twReal], %[twIndices](%[twiddleReal])\n" // twReal <- gather(@twiddleReal, vTwiddle);
    "fgw.ps     %[twImg], %[twIndices](%[twiddleImg])\n"   // twImg <- gather(@twiddleImg, vTwiddle);
    // gather xReal multiply values
    "fslli.pi   %[mulIndices], %[mulIndices], 2\n"      // mulIndices << 2
    "fgw.ps     %[xRealMul], %[mulIndices](%[XReal])\n" // xRealMul <- gather(@XReal, mulIndices);
    "fgw.ps     %[xImgMul], %[mulIndices](%[XImg])\n"   // xImgMul <- gather(@XImg, mulIndices);
    // gather xReal add/sub values
    "fslli.pi   %[addSubIndices], %[addSubIndices], 2\n"   // mulIndices << 2
    "fgw.ps     %[xRealAdd], %[addSubIndices](%[XReal])\n" // xRealAdd <- gather(@XReal, addSubIndices);
    "fgw.ps     %[xImgAdd], %[addSubIndices](%[XImg])\n"   // xImgAdd <- gather(@XImg, addSubIndices);

    // multiply op
    // real
    "fmul.ps    %[termReal], %[twImg], %[xImgMul]\n"                // (twImg * xImgMul)
    "fmsub.ps   %[termReal], %[twReal], %[xRealMul], %[termReal]\n" // termReal <- twReal * xRealMul - (twImg *
                                                                    // xImgMul);
    // img
    "fmul.ps    %[termImg], %[twReal], %[xImgMul]\n"             // (twReal * xImgMul)
    "fmadd.ps   %[termImg], %[twImg], %[xRealMul], %[termImg]\n" // termImg <- twImg * xRealMul + (twImg * xImgMul);

    // add
    "fadd.ps    %[tmp0], %[xRealAdd], %[termReal]\n" // tmp0 <- xRealAdd + termReal;
    "fadd.ps    %[tmp1], %[xImgAdd], %[termImg]\n"   // tmp1 <- xImgAdd + termImg;
    // store added elems
    "fsw.ps     %[tmp0], 0(%[resultReal])\n"
    "fsw.ps     %[tmp1], 0(%[resultImg])\n"

    // sub
    "fsub.ps    %[tmp0], %[xRealAdd], %[termReal]\n" // real <- xRealAdd - termReal;
    "fsub.ps    %[tmp1], %[xImgAdd], %[termImg]\n"   // img <- xImgAdd - termImg;
    // store sub elems
    "fsw.ps     %[tmp0], 32(%[resultReal])\n"
    "fsw.ps     %[tmp1], 32(%[resultImg])\n"

    : [ twIndices ] "=&f"(twIndices), [ mulIndices ] "=&f"(mulIndices), [ addSubIndices ] "=&f"(addSubIndices),
      [ twReal ] "=&f"(twReal), [ twImg ] "=&f"(twImg), [ xRealMul ] "=&f"(xRealMul), [ xImgMul ] "=&f"(xImgMul),
      [ xRealAdd ] "=&f"(xRealAdd), [ xImgAdd ] "=&f"(xImgAdd), [ termReal ] "=&f"(termReal),
      [ termImg ] "=&f"(termImg), [ tmp0 ] "=&f"(tmp0), [ tmp1 ] "=&f"(tmp1)
    : [ mask ] "r"(mask), [ vI ] "m"(*(const int32_t(*)[8])vI), [ round ] "r"(round), [ twiddleReal ] "r"(twiddleReal),
      [ twiddleImg ] "r"(twiddleImg), [ XReal ] "r"(XReal), [ XImg ] "r"(XImg), [ resultReal ] "r"(resultReal),
      [ resultImg ] "r"(resultImg), [ selectMultSecond ] "m"(*(const int32_t(*)[8])selectMultSecond),
      [ selectAddOrSubFirst ] "m"(*(const int32_t(*)[8])selectAddOrSubFirst)
    : "memory");
}

INLINE_ATTR void fastVectorFft16Round(const float* twiddle_real, const float* twiddle_img, float const X_real[16],
                                      const float X_img[16], int32_t round, float result_real[16], float result_img[16],
                                      const std::array<int32_t, 8>& vI, const f32x8 mulIndices,
                                      const f32x8 addsubIndices) {

  constexpr int32_t mask = 8 + 16 + 32 + 64 + 128; // 0x000000F8
  f32x8 twReal, twImg, xRealMul, xImgMul, xRealAdd, xImgAdd;
  f32x8 termReal, termImg;
  f32x8 tmp0, tmp1, twIndices;

  __asm__ __volatile__(
    // compute twiddle gather indices
    "flw.ps     %[tmp1], %[vI]\n"                      // tmp1 <- load(*vI);
    "fbcx.ps    %[twIndices], %[mask]\n"               // twIndices <- 248;
    "fbcx.ps    %[tmp0], %[round]\n"                   //
    "fsra.pi    %[twIndices], %[twIndices], %[tmp0]\n" // twIndices <- twIndices >> round;
    "fand.pi    %[twIndices], %[twIndices], %[tmp1]\n" // twIndices[i] <- twIndices[i] * i;
    // gather twiddle values
    "fslli.pi   %[twIndices], %[twIndices], 2\n"            // twIndices << 2
    "fgw.ps     %[twReal], %[twIndices](%[twiddle_real])\n" // twReal <- gather(@twiddle_real, vTwiddle);
    "fgw.ps     %[twImg], %[twIndices](%[twiddle_img])\n"   // twImg <- gather(@twiddle_img, vTwiddle);
    // gather x_real multiply values
    "fgw.ps     %[xRealMul], %[mulIndices](%[X_real])\n" // xRealMul <- gather(@X_real, mulIndices);
    "fgw.ps     %[xImgMul], %[mulIndices](%[X_img])\n"   // xImgMul <- gather(@X_img, mulIndices);
    // gather x_real add/sub values
    "fgw.ps     %[xRealAdd], %[addsubIndices](%[X_real])\n" // xRealAdd <- gather(@X_real, addsubIndices);
    "fgw.ps     %[xImgAdd], %[addsubIndices](%[X_img])\n"   // xImgAdd <- gather(@X_img, addsubIndices);

    // multiply op
    // real
    "fmul.ps    %[termReal], %[twImg], %[xImgMul]\n"                // (twImg * xImgMul)
    "fmsub.ps   %[termReal], %[twReal], %[xRealMul], %[termReal]\n" // term_real <- twReal * xRealMul - (twImg *
                                                                    // xImgMul);
    // img
    "fmul.ps    %[termImg], %[twReal], %[xImgMul]\n"             // (twReal * xImgMul)
    "fmadd.ps   %[termImg], %[twImg], %[xRealMul], %[termImg]\n" // term_img <- twImg * xRealMul + (twImg * xImgMul);

    // add
    "fadd.ps    %[tmp0], %[xRealAdd], %[termReal]\n" // tmp0 <- xRealAdd + termReal;
    "fadd.ps    %[tmp1], %[xImgAdd], %[termImg]\n"   // tmp1 <- xImgAdd + termImg;
    // store added elems
    "fsw.ps     %[tmp0], 0(%[result_real])\n"
    "fsw.ps     %[tmp1], 0(%[result_img])\n"

    // sub
    "fsub.ps    %[tmp0], %[xRealAdd], %[termReal]\n" // real <- xRealAdd - termReal;
    "fsub.ps    %[tmp1], %[xImgAdd], %[termImg]\n"   // img <- xImgAdd - termImg;
    // store sub elems
    "fsw.ps     %[tmp0], 32(%[result_real])\n"
    "fsw.ps     %[tmp1], 32(%[result_img])\n"

    : [ twIndices ] "=&f"(twIndices), [ twReal ] "=&f"(twReal), [ twImg ] "=&f"(twImg), [ xRealMul ] "=&f"(xRealMul),
      [ xImgMul ] "=&f"(xImgMul), [ xRealAdd ] "=&f"(xRealAdd), [ xImgAdd ] "=&f"(xImgAdd),
      [ termReal ] "=&f"(termReal), [ termImg ] "=&f"(termImg), [ tmp0 ] "=&f"(tmp0), [ tmp1 ] "=&f"(tmp1)
    : [ round ] "r"(round), [ mask ] "r"(mask), [ twiddle_real ] "r"(twiddle_real), [ twiddle_img ] "r"(twiddle_img),
      [ X_real ] "r"(X_real), [ X_img ] "r"(X_img), [ result_real ] "r"(result_real), [ result_img ] "r"(result_img),
      [ mulIndices ] "f"(mulIndices), [ addsubIndices ] "f"(addsubIndices), [ vI ] "m"(vI)
    : "memory");
}

INLINE_ATTR void fastVectorFft16Slice(float* real, float* img, size_t start, size_t step, [[maybe_unused]] size_t size,
                                      const float* twiddleReal, const float* twiddleImg, float resReal[16],
                                      float resImg[16], std::array<int32_t, 8> vI, const int32_t* mulIndices,
                                      const int32_t* addsubIndices, const int32_t* mulIndices2,
                                      const int32_t* addsubIndices2) {
  assert(size == 16);
  float tmp0, tmp1, mi, si;
  float tmpReal[16];
  float tmpImg[16];

  __asm__ __volatile__("flw.ps     %[mi], %[mulIndices]\n"    // mi <- load(*mulIndices);
                       "flw.ps     %[si], %[addsubIndices]\n" // si <- load(*addsubIndices);

                       "fbcx.ps    %[tmp0], %[start]\n" // tmp0 <- broadcast(start);
                       "fbcx.ps    %[tmp1], %[step]\n"  // tmp1 <- broadcast(step);

                       "fmul.pi    %[mi], %[tmp1], %[mi]\n" // mi <- vmul(step, mi);
                       "fadd.pi    %[mi], %[tmp0], %[mi]\n" // mi <- vadd(start, mi);
                       "fmul.pi    %[si], %[tmp1], %[si]\n" // si <- vmul(step, si);
                       "fadd.pi    %[si], %[tmp0], %[si]\n" // si <- vadd(start, si);

                       "fslli.pi   %[mi], %[mi], 2\n" // mi << 2 # FP 32
                       "fslli.pi   %[si], %[si], 2\n" // si << 2 # FP 32
                       : [ mi ] "=&f"(mi), [ si ] "=&f"(si), [ tmp0 ] "=&f"(tmp0), [ tmp1 ] "=&f"(tmp1)
                       : [ start ] "r"(start), [ step ] "r"(step), [ mulIndices ] "m"(*(const int32_t(*)[8])mulIndices),
                         [ addsubIndices ] "m"(*(const int32_t(*)[8])addsubIndices)
                       : "memory");

  fastVectorFft16Round(twiddleReal, twiddleImg, real, img, 0, tmpReal, tmpImg, vI, mi, si);

  __asm__ __volatile__("flw.ps     %[mi], %[mulIndices2]\n"    // mi <- load(*mulIndices2);
                       "flw.ps     %[si], %[addsubIndices2]\n" // si <- load(*addsubIndices2);
                       "fslli.pi   %[mi], %[mi], 2\n"          // mi << 2 # FP 32
                       "fslli.pi   %[si], %[si], 2\n"          // si << 2 # FP 32
                       : [ mi ] "=f"(mi), [ si ] "=f"(si)
                       : [ mulIndices2 ] "m"(*(const int32_t(*)[8])mulIndices2),
                         [ addsubIndices2 ] "m"(*(const int32_t(*)[8])addsubIndices2)
                       : "memory");

  fastVectorFft16Round(twiddleReal, twiddleImg, tmpReal, tmpImg, 1, resReal, resImg, vI, mi, si);

  fastVectorFft16Round(twiddleReal, twiddleImg, resReal, resImg, 2, tmpReal, tmpImg, vI, mi, si);

  fastVectorFft16Round(twiddleReal, twiddleImg, tmpReal, tmpImg, 3, resReal, resImg, vI, mi, si);
}

INLINE_ATTR void vectorFft16Slice(float* real, float* img, int32_t start, int32_t step, [[maybe_unused]] size_t size,
                                  const float* twiddleReal, const float* twiddleImg, float resReal[16],
                                  float resImg[16]) {

  assert(size == 16);
  float tmpReal[16];
  float tmpImg[16];

  int32_t selectMultSecond[8] = {8, 12, 10, 14, 9, 13, 11, 15};
  int32_t selectAddOrSubFirst[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  float mulIndices, addSubIndices;
  float tmp0, tmp1, tmp2, tmp3;

  __asm__ __volatile__(
    "fbcx.ps    %[tmp0], %[start]\n"                           // tmp0 <- broadcast(start);
    "fbcx.ps    %[tmp1], %[step]\n"                            // tmp1 <- broadcast(step);
    "flw.ps     %[tmp2], 0(%[mulSecond])\n"                    // tmp2 <- load(*mulSecond);
    "flw.ps     %[tmp3], 0(%[addSubFirst])\n"                  // tmp3 <- load(*addSubFirst);
    "fmul.pi    %[mulIndices], %[tmp1], %[tmp2]\n"             // mulIndices <- vmul(step, mulSecond);
    "fadd.pi    %[mulIndices], %[tmp0], %[mulIndices]\n"       // mulIndices <- vadd(start, mulSecond);
    "fmul.pi    %[addSubIndices], %[tmp1], %[tmp3]\n"          // addSubIndices <- vmul(step, addSubFirst);
    "fadd.pi    %[addSubIndices], %[tmp0], %[addSubIndices]\n" // addSubIndices <- vadd(start, addSubFirst);
    "fsw.ps     %[mulIndices], 0(%[mulSecond])\n"
    "fsw.ps     %[addSubIndices], 0(%[addSubFirst])\n"
    : [ mulIndices ] "=&f"(mulIndices), [ addSubIndices ] "=&f"(addSubIndices), [ tmp0 ] "=&f"(tmp0),
      [ tmp1 ] "=&f"(tmp1), [ tmp2 ] "=&f"(tmp2), [ tmp3 ] "=&f"(tmp3)
    : [ start ] "r"(start), [ step ] "r"(step), [ mulSecond ] "m"(*(const int32_t(*)[8])selectMultSecond),
      [ addSubFirst ] "m"(*(const int32_t(*)[8])selectAddOrSubFirst)
    : "memory");

  vectorFft16Round(twiddleReal, twiddleImg, real, img, 0, tmpReal, tmpImg, selectMultSecond, selectAddOrSubFirst);

  constexpr int32_t selectMultSecond2[8] = {1, 3, 5, 7, 9, 11, 13, 15};
  constexpr int32_t selectAddOrSubFirst2[8] = {0, 2, 4, 6, 8, 10, 12, 14};

  vectorFft16Round(twiddleReal, twiddleImg, tmpReal, tmpImg, 1, resReal, resImg, selectMultSecond2,
                   selectAddOrSubFirst2);

  vectorFft16Round(twiddleReal, twiddleImg, resReal, resImg, 2, tmpReal, tmpImg, selectMultSecond2,
                   selectAddOrSubFirst2);

  vectorFft16Round(twiddleReal, twiddleImg, tmpReal, tmpImg, 3, resReal, resImg, selectMultSecond2,
                   selectAddOrSubFirst2);
}

#endif

static void fftWithPrecomputeAndIndices(Stack& stack, const float* baseTwiddleReal, const float* baseTwiddleImg,
                                        size_t twiddleStep, const float* fft16TwiddleReal, const float* fft16TwiddleImg,
                                        float* real, float* img, size_t start, size_t step, size_t size,
                                        float* resultReal, float* resultImg, const std::array<int32_t, 8>& vI,
                                        const int32_t* mulIndices, const int32_t* addsubIndices,
                                        const int32_t* mulIndices2, const int32_t* addsubIndices2) {

  if (size == 16) {
#ifndef FFT_HOST_TEST
    fastVectorFft16Slice(real, img, start, step, size, fft16TwiddleReal, fft16TwiddleImg, resultReal, resultImg, vI,
                         mulIndices, addsubIndices, mulIndices2, addsubIndices2);
    // vectorFft16Slice(real, img, start, step, size, fft16TwiddleReal, fft16TwiddleImg, resultReal, resultImg);
    // fft16Slice(real, img, static_cast<int32_t>(start), static_cast<int32_t>(step), size, fft16TwiddleReal,
    //            fft16TwiddleImg, resultReal, resultImg);

#else
    fft16Slice(real, img, static_cast<int32_t>(start), static_cast<int32_t>(step), size, fft16TwiddleReal,
               fft16TwiddleImg, resultReal, resultImg);
#endif
  } else if (size == 1) {
    resultReal[0] = real[start];
    resultImg[0] = img[start];
  } else {
    size_t halfSize = size >> 1;
    auto saved = stack.current();
    float* tmpRealEven = stack.push<float>(halfSize);
    float* tmpImgEven = stack.push<float>(halfSize);
    fftWithPrecomputeAndIndices(stack, baseTwiddleReal, baseTwiddleImg, 2 * twiddleStep, fft16TwiddleReal,
                                fft16TwiddleImg, real, img, start, 2 * step, halfSize, tmpRealEven, tmpImgEven, vI,
                                mulIndices, addsubIndices, mulIndices2, addsubIndices2);
    float* tmpRealOdd = stack.push<float>(halfSize);
    float* tmpImgOdd = stack.push<float>(halfSize);
    fftWithPrecomputeAndIndices(stack, baseTwiddleReal, baseTwiddleImg, 2 * twiddleStep, fft16TwiddleReal,
                                fft16TwiddleImg, real, img, start + step, 2 * step, halfSize, tmpRealOdd, tmpImgOdd, vI,
                                mulIndices, addsubIndices, mulIndices2, addsubIndices2);
#ifndef FFT_HOST_TEST
    bool useVectorReduce = (halfSize & 7) == 0;
#else
    constexpr bool useVectorReduce = false;
#endif
    if (useVectorReduce) {
      vectorReduce(baseTwiddleReal, baseTwiddleImg, twiddleStep, halfSize, tmpRealEven, tmpImgEven, tmpRealOdd,
                   tmpImgOdd, resultReal, resultImg, vI);
    } else {
      reduce(baseTwiddleReal, baseTwiddleImg, twiddleStep, halfSize, tmpRealEven, tmpImgEven, tmpRealOdd, tmpImgOdd,
             resultReal, resultImg);
    }
    stack.restore(saved);
  }
}

static void fftWithPrecompute(Stack& stack, const float* baseTwiddleReal, const float* baseTwiddleImg,
                              size_t twiddleStep, const float* fft16TwiddleReal, const float* fft16TwiddleImg,
                              float* real, float* img, const size_t start, const size_t step, size_t size,
                              float* resultReal, float* resultImg) {

  // Preload indices
  constexpr std::array<int32_t, 8> i = {0, 1, 2, 3, 4, 5, 6, 7};

  constexpr int32_t mulIndices[8] = {8, 12, 10, 14, 9, 13, 11, 15};
  constexpr int32_t addsubIndices[8] = {0, 4, 2, 6, 1, 5, 3, 7};
  constexpr int32_t mulIndices2[8] = {1, 3, 5, 7, 9, 11, 13, 15};
  constexpr int32_t addsubIndices2[8] = {0, 2, 4, 6, 8, 10, 12, 14};

  // compute twiddle step here
  fftWithPrecomputeAndIndices(stack, baseTwiddleReal, baseTwiddleImg, twiddleStep, fft16TwiddleReal, fft16TwiddleImg,
                              real, img, start, step, size, resultReal, resultImg, i, mulIndices, addsubIndices,
                              mulIndices2, addsubIndices2);
}

template <bool negateInputImg, bool normalizeOutput>
void fftReversibleWithPrecompute(Stack& stack, const float* baseTwiddleReal, const float* baseTwiddleImg,
                                 size_t twiddleStep, const float fft16TwiddleReal[16], const float fft16TwiddleImg[16],
                                 float* real, float* img, size_t start, size_t step, size_t size, float* resultReal,
                                 float* resultImg) {

  auto saved = stack.current();

  // Pack the input and invert imaginary component
  float* real2;
  float* img2;
  size_t start2;
  size_t step2;
  if constexpr (negateInputImg) {
    real2 = stack.push<float>(size);
    img2 = stack.push<float>(size);
    for (size_t i = 0; i < size; ++i) {
      real2[i] = real[start + i * step];
      img2[i] = -img[start + i * step];
    }
    start2 = 0;
    step2 = 1;
  } else {
    real2 = real;
    img2 = img;
    start2 = start;
    step2 = step;
  }

  fftWithPrecompute(stack, baseTwiddleReal, baseTwiddleImg, twiddleStep, fft16TwiddleReal, fft16TwiddleImg, real2, img2,
                    start2, step2, size, resultReal, resultImg);

  if constexpr (normalizeOutput) {
    float reciprocal = rec(size);
    float minusReciprocal = -reciprocal;
    for (size_t i = 0; i < size; ++i) {
      resultReal[i] = reciprocal * resultReal[i];
      resultImg[i] = minusReciprocal * resultImg[i];
    }
  }

  stack.restore(saved);
}

template <bool inverse = false>
INLINE_ATTR void fft(size_t size, float* real, float* img, float* resultReal, float* resultImg) {

  size_t minionId = 0;
  Stack stack(minionId);
  auto saved = stack.current();

  float* baseTwiddleReal = nullptr;
  float* baseTwiddleImg = nullptr;

  twiddleVector(stack, size, baseTwiddleReal, baseTwiddleImg);

  constexpr size_t twiddleStep = 1;
  constexpr size_t start = 0;
  constexpr size_t step = 1;
  constexpr bool negateInputImg = inverse;
  constexpr bool normalizeOutput = inverse;
  fftReversibleWithPrecompute<negateInputImg, normalizeOutput>(stack, baseTwiddleReal, baseTwiddleImg, twiddleStep,
                                                               tFft16Real, tFft16Img, real, img, start, step, size,
                                                               resultReal, resultImg);
  stack.restore(saved);
}

#ifndef FFT_HOST_TEST
template <size_t fcc = 0> INLINE_ATTR void sendCredit(size_t destMinionId) {

  constexpr size_t thread = 0;
  size_t destShireId = destMinionId >> log2(SOC_MINIONS_PER_SHIRE);
  size_t destLocalMinionId = destMinionId & (SOC_MINIONS_PER_SHIRE - 1);
  size_t mask = 1 << destLocalMinionId;
  // et_printf("sendCredit: srcMId=%d dstMId=%d dstSId=%d dstLMId=%d\n", get_minion_id(), destMinionId, destShireId,
  //          destLocalMinionId);
  fcc_send(static_cast<uint32_t>(destShireId), thread, fcc, mask);
}

template <size_t fcc = 0> INLINE_ATTR void consumeCredit() {
  // size_t minionId = get_minion_id();
  // size_t shireId = minionId >> log2(SOC_MINIONS_PER_SHIRE);
  // size_t localMinionId = minionId & (SOC_MINIONS_PER_SHIRE - 1);
  // et_printf("consumeCredit: mId=%d SId=%d LMId=%d\n", minionId, shireId, localMinionId);
  fcc_consume(fcc);
}
#endif

INLINE_ATTR void barrier([[maybe_unused]] size_t globalMinionOffset, size_t range, [[maybe_unused]] size_t step) {
#ifndef FFT_HOST_TEST
  constexpr size_t fcc = 0;
  constexpr size_t thread = 0;

  size_t minionId = get_minion_id();
  size_t clippedRange = std::min(range, static_cast<size_t>(SOC_MINIONS_PER_SHIRE));
  size_t first = minionId & ~(clippedRange - 1);
  size_t firstLocal = first & (SOC_MINIONS_PER_SHIRE - 1);
  size_t log2ClippedRange = log2(clippedRange);
  size_t endLocal = firstLocal + clippedRange;
  size_t flb = first >> log2ClippedRange;
  size_t log2Range = log2(range);

  assert((endLocal & ~(SOC_MINIONS_PER_SHIRE - 1)) == 0);
  assert(isPowerOfTwo(range));

  if (range > 1) {

    // Minion synchronization (within shire)
    // et_printf("intra-shire: mid=%d range=%d clpRange=%d flb=%d end=%d step=%d\n", minionId, range, clippedRange, flb,
    // endLocal, step);
    size_t mask = 0;
    for (size_t i = firstLocal; i < endLocal; i += step) {
      mask = mask | (1 << i);
    }
    if (flbarrier(flb, clippedRange - 1)) {
      fcc_send(SHIRE_OWN, thread, fcc, mask);
    }
    fcc_consume(fcc);

    // Shire synchronization (within ETSoC)
    if (range > SOC_MINIONS_PER_SHIRE) {
      // First minion from each shire handles the ETSoC level synchronization
      if ((minionId & (SOC_MINIONS_PER_SHIRE - 1)) == 0) {
        // et_printf("etsoc-level: mid=%d range=%d flb=%d end=%d step=%d\n", minionId, range, flb, endLocal, step);
        size_t treeLevels = log2Range - log2(SOC_MINIONS_PER_SHIRE);
        size_t localShireId = (minionId - globalMinionOffset) >> log2(SOC_MINIONS_PER_SHIRE);
        // Send credit from right to left shire, in steps of 1, 2, 4... up to 2^treeLevels
        for (size_t index = 0; index < treeLevels; index++) {
          size_t bit = 1 << index;
          if ((localShireId & bit) == 0) {
            consumeCredit();
          } else {
            size_t shireMask = (1 << (index + 1)) - 1;
            size_t destShire = localShireId & ~shireMask;
            sendCredit(destShire << log2(SOC_MINIONS_PER_SHIRE));
            consumeCredit();
            break;
          }
        }
        // Wake up as many shires as trailing zeros in shireId
        size_t wakeUps = countTrailingZeros(localShireId, treeLevels);
        // et_printf("wakeups: localShireId=%d wakeUps=%d\n", localShireId, wakeUps);
        for (int index = static_cast<int>(wakeUps) - 1; index >= 0; index--) {
          size_t destShire = localShireId + (1 << index);
          sendCredit(destShire << log2(SOC_MINIONS_PER_SHIRE));
        }
        // The first minion in shire sends a credit to all minions in the shire (including itself)
        fcc_send(SHIRE_OWN, thread, fcc, mask);
      }
      // The whole shire sleeps until first minion in shire sends a credit
      fcc_consume(fcc);
    }
  }
#endif
}

template <bool negateInputImg, bool normalizeOutput>
INLINE_ATTR void fftReversibleWithPrecomputeThreaded(size_t workBranchBits, [[maybe_unused]] size_t minionOffset,
                                                     size_t minionId, Stack& stack, float* baseTwiddleReal,
                                                     float* baseTwiddleImg, const float fft16TwiddleReal[16],
                                                     const float fft16TwiddleImg[16], float* real, float* img,
                                                     size_t start, size_t step, size_t size, float* resultReal,
                                                     float* resultImg) {
  auto saved = stack.current();

  // Set start, step, size and twiddleStep for minionId
  size_t twiddleStep = 1;
  for (int index = int(workBranchBits) - 1; index >= 0; index--) {
    size_t bit = 1 << index;
    if ((minionId & bit) != 0) {
      start = start + step;
    }
    step <<= 1;
    size >>= 1;
    twiddleStep <<= 1;
  }

  // When using just one minion uset resultReal and resultImg as destination, the stack otherwise
  size_t numMinions = 1 << workBranchBits;
  float* tmpReal;
  float* tmpImg;
  [[maybe_unused]] bool isTemp;
  if (numMinions == 1) {
    tmpReal = resultReal;
    tmpImg = resultImg;
    isTemp = false;
  } else {
    tmpReal = stack.push<float>(size);
    tmpImg = stack.push<float>(size);
    isTemp = true;
  }

  // et_printf("fft_w_p: minionId=%d start=%d step=%d size=%d numMinions=%d", minionId, start, step, size, numMinions);

  // Perform the FFT recursion branch assigne to the minion
  fftReversibleWithPrecompute<negateInputImg, normalizeOutput>(stack, baseTwiddleReal, baseTwiddleImg, twiddleStep,
                                                               fft16TwiddleReal, fft16TwiddleImg, real, img, start,
                                                               step, size, tmpReal, tmpImg);

#ifndef FFT_HOST_TEST
  constexpr auto dstLevel = uint64_t(cop_dest::to_L3); // Evict all the way to L3 cache
  size_t numLinesMinusOne = ((size * sizeof(float) + CACHE_LINE_BYTES - 1) >> LOG2_CACHE_LINE_BYTES) - 1;
  fence_evict_va(0, dstLevel, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmpReal)), numLinesMinusOne,
                 CACHE_LINE_BYTES, 0);
  fence_evict_va(0, dstLevel, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmpImg)), numLinesMinusOne,
                 CACHE_LINE_BYTES, 0);
#endif

  // Now perform workBranchBits reduce passes
  //
  // - First pass with reduces pairs of consecutive elements
  // - Second pass reduces pairs of consecutive even numbers
  // - Third pass reduces pairs of consecutive multiples of 4
  // - And so on.
  //
  size_t minionStep = 1;
  assert((minionOffset & (numMinions - 1)) == 0);

  for (size_t index = 1; index <= workBranchBits; index++) {
    if ((minionId & minionStep) == 0) {
#ifndef FFT_HOST_TEST
      size_t destMinion = (get_minion_id() & ~((1 << index) - 1)) + minionStep;
      sendCredit<0>(destMinion);
      consumeCredit<1>();
#endif
      float* evenReal = tmpReal;
      float* evenImg = tmpImg;
      float* oddReal = Stack::offset(tmpReal, minionStep);
      float* oddImg = Stack::offset(tmpImg, minionStep);
      if (index == workBranchBits) {
        tmpReal = resultReal;
        tmpImg = resultImg;
      } else {
        tmpReal = stack.push<float>(2 * size);
        tmpImg = stack.push<float>(2 * size);
      }
      twiddleStep >>= 1;
#ifndef FFT_HOST_TEST
      numLinesMinusOne = ((size * sizeof(float) + CACHE_LINE_BYTES - 1) >> LOG2_CACHE_LINE_BYTES) - 1;
      fence_evict_va(0, dstLevel, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(oddReal)), numLinesMinusOne,
                     CACHE_LINE_BYTES, 0);
      fence_evict_va(0, dstLevel, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(oddImg)), numLinesMinusOne,
                     CACHE_LINE_BYTES, 0);
#endif
      reduce(baseTwiddleReal, baseTwiddleImg, twiddleStep, size, evenReal, evenImg, oddReal, oddImg, tmpReal, tmpImg);
#ifndef FFT_HOST_TEST
      numLinesMinusOne = ((2 * size * sizeof(float) + CACHE_LINE_BYTES - 1) >> LOG2_CACHE_LINE_BYTES) - 1;
      fence_evict_va(0, dstLevel, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmpReal)), numLinesMinusOne,
                     CACHE_LINE_BYTES, 0);
      fence_evict_va(0, dstLevel, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmpImg)), numLinesMinusOne,
                     CACHE_LINE_BYTES, 0);
#endif
    } else {
#ifndef FFT_HOST_TEST
      consumeCredit<0>();
      size_t destMinion = get_minion_id() & ~((1 << index) - 1);
      sendCredit<1>(destMinion);
#endif
      break;
    }
    minionStep <<= 1;
    size <<= 1;
  }

  stack.restore(saved);
}

template <bool inverse = false>
INLINE_ATTR void fftThreaded(size_t workBranchBits, [[maybe_unused]] size_t globalMinionOffset, size_t minionOffset,
                             size_t minionId, size_t size, float* real, float* img, float* resultReal,
                             float* resultImg) {
  Stack stack(minionId);
  auto saved = stack.current();

  // Preccompute twiddle vector for general FFT
  float* baseTwiddleReal = nullptr;
  float* baseTwiddleImg = nullptr;

  twiddleVector(stack, size, baseTwiddleReal, baseTwiddleImg);

  constexpr bool negateInputImg = inverse;
  constexpr bool normalizeOutput = false;
  fftReversibleWithPrecomputeThreaded<negateInputImg, normalizeOutput>(
    workBranchBits, minionOffset, minionId, stack, baseTwiddleReal, baseTwiddleImg, tFft16Real, tFft16Img, real, img, 0,
    1, size, resultReal, resultImg);
  stack.restore(saved);
}

template <bool inverse = false>
INLINE_ATTR void fft2d(size_t width, size_t height, float* real, size_t realStride, float* img, size_t imgStride,
                       float* resultReal, size_t resultRealStride, float* resultImg, size_t resultImgStride) {

  size_t minionId = 0;
  Stack stack(minionId);
  auto saved = stack.current();

  assert(realStride == imgStride);
  assert(resultRealStride == resultImgStride);

  // Precompute twiddle vector for horizontal FFT
  float* horizBaseTwiddleReal = nullptr;
  float* horizBaseTwiddleImg = nullptr;

  twiddleVector(stack, width, horizBaseTwiddleReal, horizBaseTwiddleImg);

  // Precompute twiddle vector for vertical FFT
  float* vertBaseTwiddleReal = nullptr;
  float* vertBaseTwiddleImg = nullptr;

  twiddleVector(stack, height, vertBaseTwiddleReal, vertBaseTwiddleImg);

  // Storage for one column of intermediate results
  float* resultColumnReal = stack.push<float>(height);
  float* resultColumnImg = stack.push<float>(height);

  constexpr size_t twiddleStep = 1;
  constexpr bool negateInputImg = inverse;
  constexpr bool normalizeOutput = inverse;

  // Per row FFT
  for (size_t row = 0; row < height; ++row) {
    fftReversibleWithPrecompute<negateInputImg, normalizeOutput>(
      stack, horizBaseTwiddleReal, horizBaseTwiddleImg, twiddleStep, tFft16Real, tFft16Img, real + row * realStride,
      img + row * imgStride, 0, 1, width, resultReal + row * resultRealStride, resultImg + row * resultImgStride);
  }

  // Per column FFT
  for (size_t col = 0; col < width; ++col) {
    fftReversibleWithPrecompute<negateInputImg, normalizeOutput>(
      stack, vertBaseTwiddleReal, vertBaseTwiddleImg, twiddleStep, tFft16Real, tFft16Img, resultReal, resultImg, col,
      resultRealStride, height, resultColumnReal, resultColumnImg);
    for (size_t row = 0; row < height; ++row) {
      resultReal[row * resultRealStride + col] = resultColumnReal[row];
      resultImg[row * resultImgStride + col] = resultColumnImg[row];
    }
  }

  stack.restore(saved);
}

template <bool pass1 = true, bool pass2 = true, bool inverse = false, bool freqDomainFilterFusion = false>
INLINE_ATTR void fft2dThreaded(size_t workRowBits, size_t workRowBranchBits, size_t workColBits,
                               size_t workColBranchBits, size_t globalMinionOffset, size_t minionOffset,
                               size_t minionId, size_t width, size_t height, float* real, size_t realStride, float* img,
                               size_t imgStride, float* resultReal, size_t resultRealStride, float* resultImg,
                               size_t resultImgStride, [[maybe_unused]] size_t filterIndex = 0) {

  assert(workRowBits + workRowBranchBits == workColBits + workColBranchBits);

  size_t numMinions = 1 << (workRowBits + workRowBranchBits);
  if (minionId >= numMinions) {
    return;
  }

  Stack stack(minionOffset + minionId);
  auto saved = stack.current();

  assert(realStride == imgStride);
  assert(resultRealStride == resultImgStride);

  // Precompute twiddle vector for horizontal FFT
  float* horizBaseTwiddleReal = nullptr;
  float* horizBaseTwiddleImg = nullptr;

  twiddleVector(stack, width, horizBaseTwiddleReal, horizBaseTwiddleImg);

  // Precompute twiddle vector for vertical FFT
  float* vertBaseTwiddleReal = nullptr;
  float* vertBaseTwiddleImg = nullptr;

  twiddleVector(stack, height, vertBaseTwiddleReal, vertBaseTwiddleImg);

  if constexpr (pass1) {
    // Per row FFT
    size_t rowsGroupSize = 1 << workRowBits;
    for (size_t row0 = 0; row0 < height; row0 += rowsGroupSize) {
      size_t rowMinionGroupId = (minionId & ((1 << (workRowBits + workRowBranchBits)) - 1)) >> workRowBranchBits;
      size_t row = row0 + rowMinionGroupId;
      size_t minionOffset0 = (minionOffset + minionId) & ~((1 << workRowBranchBits) - 1);
      size_t minionId0 = minionId & ((1 << workRowBranchBits) - 1);
      // et_printf("%s(%d) [mId=%d nMins=%d mOfs0=%d mId0=%d row=%d]\n", __func__, __LINE__, minionId, numMinions,
      //          minionOffset0, minionId0, row);
      constexpr bool negateInputImg = inverse;
      constexpr bool normalizeOutput = false;
      fftReversibleWithPrecomputeThreaded<negateInputImg, normalizeOutput>(
        workRowBranchBits, minionOffset0, minionId0, stack, horizBaseTwiddleReal, horizBaseTwiddleImg, tFft16Real,
        tFft16Img, real + row * realStride, img + row * imgStride, 0, 1, width, resultReal + row * resultRealStride,
        resultImg + row * resultImgStride);

#ifdef GPSDK
      hart::barrier(globalMinionOffset, numMinions);
#else
      barrier(globalMinionOffset, numMinions, 1);
#endif
    }
  }

  if constexpr (pass2) {
    // Storage for one column of input
    float* columnReal;
    float* columnImg;
    columnReal = stack.push<float>(height);
    columnImg = stack.push<float>(height);

    // Storage for one column of intermediate results
    float* resultColumnReal;
    float* resultColumnImg;
    resultColumnReal = stack.push<float>(height);
    resultColumnImg = stack.push<float>(height);

    [[maybe_unused]] float reciprocal;
    [[maybe_unused]] float minusReciprocal;
    if constexpr (inverse) {
      reciprocal = rec(width * height);
      minusReciprocal = -reciprocal;
    }

    // Per column FFT
    size_t colsGroupSize = 1 << workColBits;
    for (size_t col0 = 0; col0 < width; col0 += colsGroupSize) {
      size_t colMinionGroupId = (minionId & ((1 << (workColBits + workColBranchBits)) - 1)) >> workColBranchBits;
      size_t col = col0 + colMinionGroupId;
      size_t minionOffset0 = (minionOffset + minionId) & ~((1 << workColBranchBits) - 1);
      size_t minionId0 = minionId & ((1 << workColBranchBits) - 1);
      // et_printf("%s(%d) [mId=%d nMins=%d mOfs0=%d mId0=%d col=%d]\n", __func__, __LINE__, minionId, numMinions,
      //          minionOffset0, minionId0, col);
      for (size_t row = 0; row < height; ++row) {
#ifndef FFT_HOST_TEST
        *reinterpret_cast<uint32_t*>(&columnReal[row]) =
          atomic_load_global_32(reinterpret_cast<uint32_t*>(&resultReal[row * resultRealStride + col]));
        *reinterpret_cast<uint32_t*>(&columnImg[row]) =
          atomic_load_global_32(reinterpret_cast<uint32_t*>(&resultImg[row * resultImgStride + col]));
#else
        columnReal[row] = resultReal[row * resultRealStride + col];
        columnImg[row] = resultImg[row * resultImgStride + col];
#endif
      }
      constexpr bool negateInputImg = false;
      constexpr bool normalizeOutput = false;
      fftReversibleWithPrecomputeThreaded<negateInputImg, normalizeOutput>(
        workColBranchBits, minionOffset0, minionId0, stack, vertBaseTwiddleReal, vertBaseTwiddleImg, tFft16Real,
        tFft16Img, columnReal, columnImg, 0, 1, height, resultColumnReal, resultColumnImg);
      if ((minionId & ((1 << workColBranchBits) - 1)) == 0) {
        for (size_t row = 0; row < height; ++row) {
          float realValue = resultColumnReal[row];
          float imgValue = resultColumnImg[row];
          if constexpr (inverse) {
            realValue *= reciprocal;
            imgValue *= minusReciprocal;
          }
#ifndef FFT_HOST_TEST
          uint32_t valueReal = *reinterpret_cast<uint32_t*>(&realValue);
          uint32_t valueImg = *reinterpret_cast<uint32_t*>(&imgValue);

          if constexpr (!inverse and freqDomainFilterFusion) {
            // Fusing freqDomain filter if requested
            auto mask = dnn_lib::inlining::denoiseMask[filterIndex][row * width + col];
            valueReal *= mask;
            valueImg *= mask;
          }
          atomic_store_global_32(reinterpret_cast<uint32_t*>(&resultReal[row * resultRealStride + col]), valueReal);
          atomic_store_global_32(reinterpret_cast<uint32_t*>(&resultImg[row * resultImgStride + col]), valueImg);
#else
          resultReal[row * resultRealStride + col] = realValue;
          resultImg[row * resultImgStride + col] = imgValue;
#endif
        }
      }
#ifdef GPSDK
      hart::barrier(globalMinionOffset, numMinions);
#else
      barrier(globalMinionOffset, numMinions, 1);
#endif
    }
  }

  stack.restore(saved);
}

INLINE_ATTR void fftInv(size_t size, float* real, float* img, float* resultReal, float* resultImg) {
  constexpr bool inverse = true;
  fft<inverse>(size, real, img, resultReal, resultImg);
}

INLINE_ATTR void fft2DInv(size_t width, size_t height, float* real, size_t realStride, float* img, size_t imgStride,
                          float* resultReal, size_t resultRealStride, float* resultImg, size_t resultImgStride) {
  fft2d<true>(width, height, real, realStride, img, imgStride, resultReal, resultRealStride, resultImg,
              resultImgStride);
}

template <bool pass1 = true, bool pass2 = true>
INLINE_ATTR void fft2DInvThreaded([[maybe_unused]] size_t workRowBits, [[maybe_unused]] size_t workRowBranchBits,
                                  [[maybe_unused]] size_t workColBits, [[maybe_unused]] size_t workColBranchBits,
                                  [[maybe_unused]] size_t globalMinionOffset, [[maybe_unused]] size_t minionOffset,
                                  size_t minionId, size_t width, size_t height, float* real, size_t realStride,
                                  float* img, size_t imgStride, float* resultReal, size_t resultRealStride,
                                  float* resultImg, size_t resultImgStride) {

  fft2dThreaded<pass1, pass2, true>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, globalMinionOffset,
                                    minionOffset, minionId, width, height, real, realStride, img, imgStride, resultReal,
                                    resultRealStride, resultImg, resultImgStride);
}

} // namespace dnn_lib_v2

#endif // _FFT_V2_H_

#ifdef FFT_HOST_TEST

// Build and run a test on the host as:
//
//   cp FFT.h /tmp/fft.cpp && g++ -DFFT_HOST_TEST=1 /tmp/fft.cpp -I . -std=c++17 -g && ./a.out

#include <iostream>

void print(size_t height, size_t width, const std::string& name, float* real, size_t realStride, float* img,
           size_t imgStride) {
  std::cout << "\n" << name << "\n";
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      if (j != 0) {
        std::cout << " ; ";
      }
      std::cout << real[i * realStride + j] << "," << img[i * imgStride + j];
    }
    std::cout << "\n";
  }
}

void test1() {
  std::cout << "\n>>>> Test 1\n";

  size_t size = 32;
  float resultReal[size];

  float resultImg[size];
  float real[size];
  float img[size];
  for (size_t i = 0; i < size; ++i) {
    real[i] = i;
    img[i] = 0;
  }
  print(1, size, "Input", real, 0, img, 0);

  dnn_lib::fft(size, real, img, resultReal, resultImg);
  print(1, size, "FFT", resultReal, 0, resultImg, 0);

  float expectedResultReal[size] = {496., -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.,
                                    -16., -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.,
                                    -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.};
  float expectedResultImg[size] = {
    +0.,  +162.4507262, +80.43743187, +52.74493134, +38.627417, +29.93389459, +23.9456922,  +19.49605641,
    +16., +13.13086065, +10.69085821, +8.55217818,  +6.627417,  +4.85354694,  +3.18259788,  +1.57586245,
    +0.,  -1.57586245,  -3.18259788,  -4.85354694,  -6.627417,  -8.55217818,  -10.69085821, -13.13086065,
    -16., -19.49605641, -23.9456922,  -29.93389459, -38.627417, -52.74493134, -80.43743187, -162.4507262};
  for (size_t i = 0; i < size; ++i) {
    assert(std::abs(resultReal[i] - expectedResultReal[i]) < 0.01f);
    assert(std::abs(resultImg[i] - expectedResultImg[i]) < 0.01f);
  }

  float reconstReal[size];
  float reconstImg[size];
  dnn_lib::fftInv(size, resultReal, resultImg, reconstReal, reconstImg);
  print(1, size, "Reconstructed", reconstReal, 0, reconstImg, 0);

  for (size_t i = 0; i < size; ++i) {
    assert(std::abs(reconstReal[i] - real[i]) < 0.001f);
    assert(std::abs(reconstImg[i] - img[i]) < 0.001f);
  }
}

void test2() {

  std::cout << "\n>>>> Test 2\n";

  constexpr size_t height = 2;
  constexpr size_t width = 4;

  constexpr size_t realStride = 4;
  constexpr size_t imgStride = 4;
  float real[height][realStride] = {{1, 3, 2, 1}, {1, 2, 2, 0}};
  float img[height][imgStride] = {0};
  print(height, width, "Input", &real[0][0], realStride, &img[0][0], imgStride);

  constexpr size_t resultRealStride = 4;
  constexpr size_t resultImgStride = 4;
  float resultReal[height][resultRealStride] = {0};
  float resultImg[height][imgStride] = {0};
  dnn_lib::fft2d(width, height, &real[0][0], realStride, &img[0][0], imgStride, &resultReal[0][0], resultRealStride,
                 &resultImg[0][0], resultImgStride);
  print(height, width, "FFT 2D", &resultReal[0][0], resultRealStride, &resultImg[0][0], resultImgStride);

  float expectedResultReal[height][resultRealStride] = {{12., -2., 0., -2.}, {2., 0., -2., 0.}};
  float expectedResultImg[height][imgStride] = {{0., -4., 0., 4.}, {0., 0., 0., 0.}};
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      assert(std::abs(resultReal[row][col] - expectedResultReal[row][col]) < 0.00001f);
      assert(std::abs(resultImg[row][col] - expectedResultImg[row][col]) < 0.00001f);
    }
  }

  std::cout << "Reconstructed\n";

  constexpr size_t reconstRealStride = 4;
  constexpr size_t reconstImgStride = 4;
  float reconstReal[height][reconstRealStride] = {0};
  float reconstImg[height][reconstImgStride] = {0};

  dnn_lib::fft2DInv(width, height, &resultReal[0][0], resultRealStride, &resultImg[0][0], resultImgStride,
                    &reconstReal[0][0], reconstRealStride, &reconstImg[0][0], reconstImgStride);
  print(height, width, "Reconstructed", &reconstReal[0][0], reconstRealStride, &reconstImg[0][0], reconstImgStride);

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      assert(std::abs(reconstReal[row][col] - real[row][col]) < 0.000001f);
      assert(std::abs(reconstImg[row][col] - img[row][col]) < 0.000001f);
    }
  }
}

void test3() {

  std::cout << "\n>>>> Test 3\n";

  constexpr size_t height = 1;
  constexpr size_t width = 1;

  constexpr size_t realStride = width;
  constexpr size_t imgStride = width;

  float real[height][realStride];
  float img[height][imgStride];
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      real[row][col] = row;
      img[row][col] = col;
    }
  }
  print(height, width, "Input", &real[0][0], realStride, &img[0][0], imgStride);

  constexpr size_t resultRealStride = width;
  constexpr size_t resultImgStride = width;
  float resultReal[height][resultRealStride] = {0};
  float resultImg[height][imgStride] = {0};
  dnn_lib::fft2d(width, height, &real[0][0], realStride, &img[0][0], imgStride, &resultReal[0][0], resultRealStride,
                 &resultImg[0][0], resultImgStride);
  print(height, width, "FFT 2D", &resultReal[0][0], resultRealStride, &resultImg[0][0], resultImgStride);
}

void test4() {

  std::cout << "\n>>>> Test 4\n";

  size_t size = 32;
  float resultReal[size];

  float resultImg[size];
  float real[size];
  float img[size];
  for (size_t i = 0; i < size; ++i) {
    real[i] = i;
    img[i] = 0;
  }
  print(1, size, "Input", real, 0, img, 0);

  constexpr size_t workBranchBits = 2;
  constexpr size_t globalMinionOffset = 32;
  constexpr size_t minionOffset = 16;
  //
  // Running the threaded FFT with a sequence of minion ids that ensures
  // the final FFT result is correct, irrespective of the fact that the
  // barrier mechanism for syncrhonizing before the reduce operations may
  // be broken.
  //
  // Firstly, minions 1 and 3 populate their stacks because their computations
  // do not depend on the stack from any other minion. These two runs abort
  // before writting anything to the destination vector, but they have the side
  // effect of populating the stacks for minion 1 and 3 with valid intermediate
  // results.
  //
  // Then minion 2 runs because it only depends on minion 3, but again it does
  // not write any result into the destination vectors. Finally, minion 0 that
  // depends on minion 2 can run and leaves the correct result on the
  // destination vectors.
  //
  dnn_lib::fftThreaded(workBranchBits, globalMinionOffset, minionOffset, 1, size, real, img, resultReal, resultImg);
  dnn_lib::fftThreaded(workBranchBits, globalMinionOffset, minionOffset, 3, size, real, img, resultReal, resultImg);
  dnn_lib::fftThreaded(workBranchBits, globalMinionOffset, minionOffset, 2, size, real, img, resultReal, resultImg);
  dnn_lib::fftThreaded(workBranchBits, globalMinionOffset, minionOffset, 0, size, real, img, resultReal, resultImg);

  float expectedResultReal[size] = {496., -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.,
                                    -16., -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.,
                                    -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.};
  float expectedResultImg[size] = {
    +0.,  +162.4507262, +80.43743187, +52.74493134, +38.627417, +29.93389459, +23.9456922,  +19.49605641,
    +16., +13.13086065, +10.69085821, +8.55217818,  +6.627417,  +4.85354694,  +3.18259788,  +1.57586245,
    +0.,  -1.57586245,  -3.18259788,  -4.85354694,  -6.627417,  -8.55217818,  -10.69085821, -13.13086065,
    -16., -19.49605641, -23.9456922,  -29.93389459, -38.627417, -52.74493134, -80.43743187, -162.4507262};
  for (size_t i = 0; i < size; ++i) {
    assert(std::abs(resultReal[i] - expectedResultReal[i]) < 0.01f);
    assert(std::abs(resultImg[i] - expectedResultImg[i]) < 0.01f);
  }

  float reconstReal[size];
  float reconstImg[size];
  dnn_lib::fftInv(size, resultReal, resultImg, reconstReal, reconstImg);
  print(1, size, "Reconstructed", reconstReal, 0, reconstImg, 0);

  for (size_t i = 0; i < size; ++i) {
    assert(std::abs(reconstReal[i] - real[i]) < 0.001f);
    assert(std::abs(reconstImg[i] - img[i]) < 0.001f);
  }
}

void test5() {

  std::cout << "\n>>>> Test 5\n";

  constexpr size_t height = 2;
  constexpr size_t width = 4;

  constexpr size_t realStride = 4;
  constexpr size_t imgStride = 4;
  float real[height][realStride] = {{1, 3, 2, 1}, {1, 2, 2, 0}};
  float img[height][imgStride] = {0};
  print(height, width, "Input", &real[0][0], realStride, &img[0][0], imgStride);

  size_t workRowBits = 1;
  size_t workRowBranchBits = 1;
  size_t workColBits = 2;
  size_t workColBranchBits = 0;
  constexpr size_t globalMinionOffset = 32;
  constexpr size_t minionOffset = 16;
  constexpr size_t resultRealStride = 4;
  constexpr size_t resultImgStride = 4;
  float resultReal[height][resultRealStride] = {0};
  float resultImg[height][imgStride] = {0};
  dnn_lib::fft2dThreaded<true, false>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                      globalMinionOffset, minionOffset, 3, width, height, &real[0][0], realStride,
                                      &img[0][0], imgStride, &resultReal[0][0], resultRealStride, &resultImg[0][0],
                                      resultImgStride);
  dnn_lib::fft2dThreaded<true, false>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                      globalMinionOffset, minionOffset, 2, width, height, &real[0][0], realStride,
                                      &img[0][0], imgStride, &resultReal[0][0], resultRealStride, &resultImg[0][0],
                                      resultImgStride);
  dnn_lib::fft2dThreaded<true, false>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                      globalMinionOffset, minionOffset, 1, width, height, &real[0][0], realStride,
                                      &img[0][0], imgStride, &resultReal[0][0], resultRealStride, &resultImg[0][0],
                                      resultImgStride);
  dnn_lib::fft2dThreaded<true, false>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                      globalMinionOffset, minionOffset, 0, width, height, &real[0][0], realStride,
                                      &img[0][0], imgStride, &resultReal[0][0], resultRealStride, &resultImg[0][0],
                                      resultImgStride);
  dnn_lib::fft2dThreaded<false, true>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                      globalMinionOffset, minionOffset, 0, width, height, &real[0][0], realStride,
                                      &img[0][0], imgStride, &resultReal[0][0], resultRealStride, &resultImg[0][0],
                                      resultImgStride);
  dnn_lib::fft2dThreaded<false, true>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                      globalMinionOffset, minionOffset, 1, width, height, &real[0][0], realStride,
                                      &img[0][0], imgStride, &resultReal[0][0], resultRealStride, &resultImg[0][0],
                                      resultImgStride);
  dnn_lib::fft2dThreaded<false, true>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                      globalMinionOffset, minionOffset, 2, width, height, &real[0][0], realStride,
                                      &img[0][0], imgStride, &resultReal[0][0], resultRealStride, &resultImg[0][0],
                                      resultImgStride);
  dnn_lib::fft2dThreaded<false, true>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                      globalMinionOffset, minionOffset, 3, width, height, &real[0][0], realStride,
                                      &img[0][0], imgStride, &resultReal[0][0], resultRealStride, &resultImg[0][0],
                                      resultImgStride);
  print(height, width, "FFT 2D", &resultReal[0][0], resultRealStride, &resultImg[0][0], resultImgStride);

  float expectedResultReal[height][resultRealStride] = {{12., -2., 0., -2.}, {2., 0., -2., 0.}};
  float expectedResultImg[height][imgStride] = {{0., -4., 0., 4.}, {0., 0., 0., 0.}};
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      assert(std::abs(resultReal[row][col] - expectedResultReal[row][col]) < 0.00001f);
      assert(std::abs(resultImg[row][col] - expectedResultImg[row][col]) < 0.00001f);
    }
  }

  constexpr size_t reconstRealStride = 4;
  constexpr size_t reconstImgStride = 4;
  float reconstReal[height][reconstRealStride] = {0};
  float reconstImg[height][reconstImgStride] = {0};
  constexpr bool inverse = true;
  dnn_lib::fft2dThreaded<true, false, inverse>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                               globalMinionOffset, minionOffset, 3, width, height, &resultReal[0][0],
                                               resultRealStride, &resultImg[0][0], resultImgStride, &reconstReal[0][0],
                                               reconstRealStride, &reconstImg[0][0], reconstImgStride);
  dnn_lib::fft2dThreaded<true, false, inverse>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                               globalMinionOffset, minionOffset, 2, width, height, &resultReal[0][0],
                                               resultRealStride, &resultImg[0][0], resultImgStride, &reconstReal[0][0],
                                               reconstRealStride, &reconstImg[0][0], reconstImgStride);
  dnn_lib::fft2dThreaded<true, false, inverse>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                               globalMinionOffset, minionOffset, 1, width, height, &resultReal[0][0],
                                               resultRealStride, &resultImg[0][0], resultImgStride, &reconstReal[0][0],
                                               reconstRealStride, &reconstImg[0][0], reconstImgStride);
  dnn_lib::fft2dThreaded<true, false, inverse>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                               globalMinionOffset, minionOffset, 0, width, height, &resultReal[0][0],
                                               resultRealStride, &resultImg[0][0], resultImgStride, &reconstReal[0][0],
                                               reconstRealStride, &reconstImg[0][0], reconstImgStride);
  dnn_lib::fft2dThreaded<false, true, inverse>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                               globalMinionOffset, minionOffset, 0, width, height, &resultReal[0][0],
                                               resultRealStride, &resultImg[0][0], resultImgStride, &reconstReal[0][0],
                                               reconstRealStride, &reconstImg[0][0], reconstImgStride);
  dnn_lib::fft2dThreaded<false, true, inverse>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                               globalMinionOffset, minionOffset, 1, width, height, &resultReal[0][0],
                                               resultRealStride, &resultImg[0][0], resultImgStride, &reconstReal[0][0],
                                               reconstRealStride, &reconstImg[0][0], reconstImgStride);
  dnn_lib::fft2dThreaded<false, true, inverse>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                               globalMinionOffset, minionOffset, 2, width, height, &resultReal[0][0],
                                               resultRealStride, &resultImg[0][0], resultImgStride, &reconstReal[0][0],
                                               reconstRealStride, &reconstImg[0][0], reconstImgStride);
  dnn_lib::fft2dThreaded<false, true, inverse>(workRowBits, workRowBranchBits, workColBits, workColBranchBits,
                                               globalMinionOffset, minionOffset, 3, width, height, &resultReal[0][0],
                                               resultRealStride, &resultImg[0][0], resultImgStride, &reconstReal[0][0],
                                               reconstRealStride, &reconstImg[0][0], reconstImgStride);
  print(height, width, "Reconstructed", &reconstReal[0][0], reconstRealStride, &reconstImg[0][0], reconstImgStride);

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      assert(std::abs(reconstReal[row][col] - real[row][col]) < 0.000001f);
      assert(std::abs(reconstImg[row][col] - img[row][col]) < 0.000001f);
    }
  }
}

void test6() {

  std::cout << "\n>>>> Test 6 sim size 16x16 fft2d \n";

  constexpr size_t height = 8;
  constexpr size_t width = 8;

  constexpr size_t realStride = width;
  constexpr size_t imgStride = width;

  float real[height][realStride];
  float img[height][imgStride];
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      real[row][col] = row;
      img[row][col] = col;
    }
  }
  // print(height, width, "Input", &real[0][0], realStride, &img[0][0], imgStride);

  constexpr size_t resultRealStride = width;
  constexpr size_t resultImgStride = width;
  float resultReal[height][resultRealStride] = {0};
  float resultImg[height][imgStride] = {0};
  dnn_lib::fft2d(width, height, &real[0][0], realStride, &img[0][0], imgStride, &resultReal[0][0], resultRealStride,
                 &resultImg[0][0], resultImgStride);
  // print(height, width, "FFT 2D", &resultReal[0][0], resultRealStride, &resultImg[0][0], resultImgStride);
}

void test7() {

  std::cout << "\n>>>> Test 7 sim size 256x256 fft2d \n";

  constexpr size_t height = 256;
  constexpr size_t width = 256;

  constexpr size_t realStride = width;
  constexpr size_t imgStride = width;

  float real[height][realStride];
  float img[height][imgStride];
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      real[row][col] = row;
      img[row][col] = col;
    }
  }
  // uncomment print in case to show the 256 precalculated values for 256X256 input images
  // print(height, width, "Input", &real[0][0], realStride, &img[0][0], imgStride);

  constexpr size_t resultRealStride = width;
  constexpr size_t resultImgStride = width;
  float resultReal[height][resultRealStride] = {0};
  float resultImg[height][imgStride] = {0};
  dnn_lib::fft2d(width, height, &real[0][0], realStride, &img[0][0], imgStride, &resultReal[0][0], resultRealStride,
                 &resultImg[0][0], resultImgStride);
  // print(height, width, "FFT 2D", &resultReal[0][0], resultRealStride, &resultImg[0][0], resultImgStride);
}

int main(int argc, char** argv) {

  test1();
  test2();
  test3();
  test4();
  test5();
  test6();
  test7();
  std::cout << "\nALL PASSED\n\n";

  return 0;
}

#endif
