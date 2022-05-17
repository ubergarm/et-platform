#ifndef _FFT_H_
#define _FFT_H_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#ifndef INLINE_ATTR
#define INLINE_ATTR __attribute__((always_inline)) inline
#endif

namespace dnn_lib {

// Prototypes

template <size_t workBranchBits = 0>
INLINE_ATTR void fft(size_t size, float* real, float* img, float* result_real, float* result_img);

INLINE_ATTR void fft2d(size_t width, size_t height, float* real, size_t real_stride, float* img, size_t img_stride,
                       float* result_real, size_t result_real_stride, float* result_img, size_t result_img_stride);

INLINE_ATTR void fft_inv(size_t size, float* real, float* img, float* result_real, float* result_img);

INLINE_ATTR void fft2d_inv(size_t width, size_t height, float* real, size_t real_stride, float* img, size_t img_stride,
                           float* result_real, size_t result_real_stride, float* result_img, size_t result_img_stride);

// Implementation

class Stack {
private:
  using EType = uint32_t;
  static constexpr size_t elementsPerMinion = 16 * 1024;
  static constexpr size_t numMinions = 1024;

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
    constexpr size_t baseElements = (sizeof(T) * elements + sizeof(EType) - 1) / sizeof(EType);
    pointer += baseElements;
    return result;
  }

  template <typename T> T* push(size_t elements) {
    T* result = reinterpret_cast<T*>(pointer);
    size_t baseElements = (sizeof(T) * elements + sizeof(EType) - 1) / sizeof(EType);
    pointer += baseElements;
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

INLINE_ATTR void euler_formula(float angle, float& real, float& img) {
  real = cos(angle);
  img = sin(angle);
}

INLINE_ATTR void w(size_t j, float recN, float& real, float& img) {
  return euler_formula(-2.f * static_cast<float>(M_PI) * static_cast<float>(j) * recN, real, img);
}

INLINE_ATTR void twiddle_vector_small(size_t size, float real[], float img[]) {
  float recN = rec(size);
  for (size_t i = 0; i < size; ++i) {
    w(i, recN, real[i], img[i]);
  }
}

INLINE_ATTR void twiddle_vector_big(size_t n, float real[], float img[]) {

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

#ifndef FFT_HOST_TEST

INLINE_ATTR void vector_fft16_round(float twiddle_real[16], float twiddle_img[16], float X_real[16], float X_img[16],
                                    int32_t round, float result_real[16], float result_img[16],
                                    const int32_t select_mult_second[8], const int32_t select_add_or_sub_first[8]) {

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
    "flw.ps     %[tmp1], %[vI]\n"
    "fand.pi    %[twIndices], %[twIndices], %[tmp1]\n" // twIndices[i] <- twIndices[i] * i;
    // load multiply, add, sub gather indices
    "flw.ps     %[mulIndices], %[select_mult_second]\n"         // mulIndices <- load(@select_mult_second, 8);
    "flw.ps     %[addSubIndices], %[select_add_or_sub_first]\n" // addSubIndices <- load(@select_add_or_sub_first, 8);
    // gather twiddle values
    "fslli.pi   %[twIndices], %[twIndices], 2\n"            // twIndices << 2
    "fgw.ps     %[twReal], %[twIndices](%[twiddle_real])\n" // twReal <- gather(@twiddle_real, vTwiddle);
    "fgw.ps     %[twImg], %[twIndices](%[twiddle_img])\n"   // twImg <- gather(@twiddle_img, vTwiddle);
    // gather x_real multiply values
    "fslli.pi   %[mulIndices], %[mulIndices], 2\n"       // mulIndices << 2
    "fgw.ps     %[xRealMul], %[mulIndices](%[X_real])\n" // xRealMul <- gather(@X_real, mulIndices);
    "fgw.ps     %[xImgMul], %[mulIndices](%[X_img])\n"   // xImgMul <- gather(@X_img, mulIndices);
    // gather x_real add/sub values
    "fslli.pi   %[addSubIndices], %[addSubIndices], 2\n"    // mulIndices << 2
    "fgw.ps     %[xRealAdd], %[addSubIndices](%[X_real])\n" // xRealAdd <- gather(@X_real, addSubIndices);
    "fgw.ps     %[xImgAdd], %[addSubIndices](%[X_img])\n"   // xImgAdd <- gather(@X_img, addSubIndices);

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
    "fsw.ps     %[tmp0], (%[result_real])\n"
    "fsw.ps     %[tmp1], (%[result_img])\n"

    // sub
    "fsub.ps    %[tmp0], %[xRealAdd], %[termReal]\n" // real <- xRealAdd - termReal;
    "fsub.ps    %[tmp1], %[xImgAdd], %[termImg]\n"   // img <- xImgAdd - termImg;
    // store sub elems
    "fsw.ps     %[tmp0], 32(%[result_real])\n"
    "fsw.ps     %[tmp1], 32(%[result_img])\n"

    : [ twIndices ] "=&f"(twIndices), [ mulIndices ] "=&f"(mulIndices), [ addSubIndices ] "=&f"(addSubIndices),
      [ twReal ] "=&f"(twReal), [ twImg ] "=&f"(twImg), [ xRealMul ] "=&f"(xRealMul), [ xImgMul ] "=&f"(xImgMul),
      [ xRealAdd ] "=&f"(xRealAdd), [ xImgAdd ] "=&f"(xImgAdd), [ termReal ] "=&f"(termReal),
      [ termImg ] "=&f"(termImg), [ tmp0 ] "=&f"(tmp0), [ tmp1 ] "=&f"(tmp1)
    : [ mask ] "r"(mask), [ vI ] "m"(*(const int32_t(*)[8])vI), [ round ] "r"(round),
      [ twiddle_real ] "r"(twiddle_real), [ twiddle_img ] "r"(twiddle_img), [ X_real ] "r"(X_real),
      [ X_img ] "r"(X_img), [ result_real ] "r"(result_real), [ result_img ] "r"(result_img),
      [ select_mult_second ] "m"(*(const int32_t(*)[8])select_mult_second),
      [ select_add_or_sub_first ] "m"(*(const int32_t(*)[8])select_add_or_sub_first)
    : "memory");
}

INLINE_ATTR void vector_fft16_slice(float* real, float* img, int32_t start, int32_t step, size_t size,
                                    float twiddle_real[16], float twiddle_img[16], float res_real[16],
                                    float res_img[16]) {
  assert(size == 16);
  float tmp_real[16];
  float tmp_img[16];

  int32_t select_mult_second[8] = {8, 12, 10, 14, 9, 13, 11, 15};
  int32_t select_add_or_sub_first[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  float mulIndices, addsubIndices;
  float tmp0, tmp1, tmp2, tmp3;

  __asm__ __volatile__(
    "fbcx.ps    %[tmp0], %[start]\n"                           // tmp0 <- broadcast(start);
    "fbcx.ps    %[tmp1], %[step]\n"                            // tmp1 <- broadcast(step);
    "flw.ps     %[tmp2], %[mul_second]\n"                      // tmp2 <- load(*mul_second);
    "flw.ps     %[tmp3], %[addsub_first]\n"                    // tmp3 <- load(*addsub_first);
    "fmul.pi    %[mulIndices], %[tmp1], %[tmp2]\n"             // mulIndices <- vmul(step, mul_second);
    "fadd.pi    %[mulIndices], %[tmp0], %[mulIndices]\n"       // mulIndices <- vadd(start, mul_second);
    "fmul.pi    %[addsubIndices], %[tmp1], %[tmp3]\n"          // addsubIndices <- vmul(step, addsub_first);
    "fadd.pi    %[addsubIndices], %[tmp0], %[addsubIndices]\n" // addsubIndices <- vadd(start, addsub_first);
    "fsw.ps     %[mulIndices], %[mul_second]\n"
    "fsw.ps     %[addsubIndices], %[addsub_first]\n"
    : [ mulIndices ] "=&f"(mulIndices), [ addsubIndices ] "=&f"(addsubIndices), [ tmp0 ] "=&f"(tmp0),
      [ tmp1 ] "=&f"(tmp1), [ tmp2 ] "=&f"(tmp2), [ tmp3 ] "=&f"(tmp3)
    : [ start ] "r"(start), [ step ] "r"(step), [ mul_second ] "m"(*(const int32_t(*)[8])select_mult_second),
      [ addsub_first ] "m"(*(const int32_t(*)[8])select_add_or_sub_first)
    : "memory");
  // TODO: pass vector register as a parameter instead of dumping back to memory
  vector_fft16_round(twiddle_real, twiddle_img, real, img, 0, tmp_real, tmp_img, select_mult_second,
                     select_add_or_sub_first);

  constexpr int32_t select_mult_second2[8] = {1, 3, 5, 7, 9, 11, 13, 15};
  constexpr int32_t select_add_or_sub_first2[8] = {0, 2, 4, 6, 8, 10, 12, 14};

  vector_fft16_round(twiddle_real, twiddle_img, tmp_real, tmp_img, 1, res_real, res_img, select_mult_second2,
                     select_add_or_sub_first2);

  vector_fft16_round(twiddle_real, twiddle_img, res_real, res_img, 2, tmp_real, tmp_img, select_mult_second2,
                     select_add_or_sub_first2);

  vector_fft16_round(twiddle_real, twiddle_img, tmp_real, tmp_img, 3, res_real, res_img, select_mult_second2,
                     select_add_or_sub_first2);
}

#endif

constexpr size_t twiddle_index(size_t round, size_t i) {
  return i & ((8 + 16 + 32 + 64 + 128) >> round);
}

INLINE_ATTR void fft16_round(float twiddle_real[16], float twiddle_img[16], float X_real[16], float X_img[16],
                             int32_t round, float result_real[16], float result_img[16],
                             const int32_t select_mult_second[8], const int32_t select_add_or_sub_first[8]) {

  // This loop is suitable for vectorizing
  for (size_t i = 0; i < 8; ++i) {
    size_t tidx = twiddle_index(round, i);
    size_t sidx = select_mult_second[i];
    size_t fidx = select_add_or_sub_first[i];
    float term_real, term_img;
    mult(twiddle_real[tidx], twiddle_img[tidx], X_real[sidx], X_img[sidx], term_real, term_img);
    add(X_real[fidx], X_img[fidx], term_real, term_img, result_real[i], result_img[i]);
    sub(X_real[fidx], X_img[fidx], term_real, term_img, result_real[i + 8], result_img[i + 8]);
  }
}

INLINE_ATTR void fft16_slice(float* real, float* img, int32_t start, int32_t step, size_t size, float twiddle_real[16],
                             float twiddle_img[16], float res_real[16], float res_img[16]) {

  assert(size == 16);
  float tmp_real[16];
  float tmp_img[16];

  int32_t select_mult_second[8] = {8, 12, 10, 14, 9, 13, 11, 15};
  int32_t select_add_or_sub_first[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  // This loop is suitable for vectorizing
  for (size_t i = 0; i < 8; ++i) {
    select_mult_second[i] = start + step * select_mult_second[i];
    select_add_or_sub_first[i] = start + step * select_add_or_sub_first[i];
  }

  fft16_round(twiddle_real, twiddle_img, real, img, 0, tmp_real, tmp_img, select_mult_second, select_add_or_sub_first);

  constexpr int32_t select_mult_second2[8] = {1, 3, 5, 7, 9, 11, 13, 15};
  constexpr int32_t select_add_or_sub_first2[8] = {0, 2, 4, 6, 8, 10, 12, 14};

  fft16_round(twiddle_real, twiddle_img, tmp_real, tmp_img, 1, res_real, res_img, select_mult_second2,
              select_add_or_sub_first2);

  fft16_round(twiddle_real, twiddle_img, res_real, res_img, 2, tmp_real, tmp_img, select_mult_second2,
              select_add_or_sub_first2);

  fft16_round(twiddle_real, twiddle_img, tmp_real, tmp_img, 3, res_real, res_img, select_mult_second2,
              select_add_or_sub_first2);
}

INLINE_ATTR void reduce(float* base_twiddle_real, float* base_twiddle_img, size_t twiddle_step, size_t half_size,
                        float* tmp_real_even, float* tmp_img_even, float* tmp_real_odd, float* tmp_img_odd,
                        float* result_real, float* result_img) {
  size_t twiddle_index = 0;
  for (size_t j = 0; j < half_size; ++j) {
    float twiddle_real = base_twiddle_real[twiddle_index];
    float twiddle_img = base_twiddle_img[twiddle_index];
    float term_real, term_img;
    mult(twiddle_real, twiddle_img, tmp_real_odd[j], tmp_img_odd[j], term_real, term_img);
    add(tmp_real_even[j], tmp_img_even[j], term_real, term_img, result_real[j], result_img[j]);
    sub(tmp_real_even[j], tmp_img_even[j], term_real, term_img, result_real[j + (half_size)],
        result_img[j + (half_size)]);
    twiddle_index += twiddle_step;
  }
}

void fft_with_precompute(Stack& stack, float* base_twiddle_real, float* base_twiddle_img, size_t twiddle_step,
                         float fft16_twiddle_real[16], float fft16_twiddle_img[16], float* real, float* img,
                         size_t start, size_t step, size_t size, float* result_real, float* result_img);

void fft_with_precompute(Stack& stack, float* base_twiddle_real, float* base_twiddle_img, size_t twiddle_step,
                         float fft16_twiddle_real[16], float fft16_twiddle_img[16], float* real, float* img,
                         size_t start, size_t step, size_t size, float* result_real, float* result_img) {
  if (size == 16) {
#ifndef FFT_HOST_TEST
    vector_fft16_slice(real, img, start, step, size, fft16_twiddle_real, fft16_twiddle_img, result_real, result_img);
#else
    fft16_slice(real, img, start, step, size, fft16_twiddle_real, fft16_twiddle_img, result_real, result_img);
#endif
  } else if (size == 1) {
    result_real[0] = real[start];
    result_img[0] = img[start];
  } else {
    size_t half_size = size >> 1;
    auto saved = stack.current();
    float* tmp_real_even = stack.push<float>(half_size);
    float* tmp_img_even = stack.push<float>(half_size);
    fft_with_precompute(stack, base_twiddle_real, base_twiddle_img, 2 * twiddle_step, fft16_twiddle_real,
                        fft16_twiddle_img, real, img, start, 2 * step, half_size, tmp_real_even, tmp_img_even);
    float* tmp_real_odd = stack.push<float>(half_size);
    float* tmp_img_odd = stack.push<float>(half_size);
    fft_with_precompute(stack, base_twiddle_real, base_twiddle_img, 2 * twiddle_step, fft16_twiddle_real,
                        fft16_twiddle_img, real, img, start + step, 2 * step, half_size, tmp_real_odd, tmp_img_odd);
    reduce(base_twiddle_real, base_twiddle_img, twiddle_step, half_size, tmp_real_even, tmp_img_even, tmp_real_odd,
           tmp_img_odd, result_real, result_img);
    stack.restore(saved);
  }
}

INLINE_ATTR void fft(size_t size, float* real, float* img, float* result_real, float* result_img) {

  size_t minionId = 0;
  Stack stack(minionId);
  auto saved = stack.current();

  float* base_twiddle_real = stack.push<float>(size);
  float* base_twiddle_img = stack.push<float>(size);
  if (size >= 16)
    twiddle_vector_big(size, base_twiddle_real, base_twiddle_img);
  else
    twiddle_vector_small(size, base_twiddle_real, base_twiddle_img);

  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);

  constexpr size_t twiddle_step = 1;
  size_t start = 0;
  size_t step = 1;
  fft_with_precompute(stack, base_twiddle_real, base_twiddle_img, twiddle_step, fft16_twiddle_real, fft16_twiddle_img,
                      real, img, start, step, size, result_real, result_img);

  stack.restore(saved);
}

INLINE_ATTR void barrier(size_t range, [[maybe_unused]] size_t step) {
#ifndef FFT_HOST_TEST
  constexpr size_t fcc = 0;
  constexpr size_t thread = 0;

  size_t minionId = get_minion_id();
  size_t first = minionId & ~(range - 1);
  size_t firstLocal = first & (SOC_MINIONS_PER_SHIRE - 1);
  size_t endLocal = firstLocal + range;
  size_t flb = firstLocal >> log2(range);

  et_printf("barrier: mid=%d range=%d flb=%d end=%d step=%d\n", minionId, range, flb, endLocal, step);

  if (flbarrier(flb, range - 1)) {
    // size_t minionId = get_minion_id();
    assert((endLocal & ~(SOC_MINIONS_PER_SHIRE - 1)) == 0);
    size_t mask = 0;
    for (size_t i = firstLocal; i < endLocal; i += step) {
      mask = mask | (1 << i);
    }
    fcc_send(SHIRE_OWN, thread, fcc, mask);
  }
  fcc_consume(fcc);
#endif
}

INLINE_ATTR void fft_threaded_with_precompute(size_t workBranchBits, size_t minionOffset, size_t minionId, Stack& stack,
                                              float* base_twiddle_real, float* base_twiddle_img,
                                              float fft16_twiddle_real[16], float fft16_twiddle_img[16], float* real,
                                              float* img, size_t start, size_t step, size_t size, float* result_real,
                                              float* result_img) {
  auto saved = stack.current();

  // Set start, step, size and twiddle_step for minionId
  size_t twiddle_step = 1;
  for (int index = int(workBranchBits) - 1; index >= 0; index--) {
    size_t bit = 1 << index;
    if ((minionId & bit) != 0) {
      start = start + step;
    }
    step <<= 1;
    size >>= 1;
    twiddle_step <<= 1;
  }

  // When using just one minion uset result_real and result_img as destination, the stack otherwise
  size_t numMinions = 1 << workBranchBits;
  float* tmp_real;
  float* tmp_img;
  [[maybe_unused]] bool isTemp;
  if (numMinions == 1) {
    tmp_real = result_real;
    tmp_img = result_img;
    isTemp = false;
  } else {
    tmp_real = stack.push<float>(size);
    tmp_img = stack.push<float>(size);
    isTemp = true;
  }

  // Perform the FFT recursion branch assigne to the minion
  fft_with_precompute(stack, base_twiddle_real, base_twiddle_img, twiddle_step, fft16_twiddle_real, fft16_twiddle_img,
                      real, img, start, step, size, tmp_real, tmp_img);

#ifndef FFT_HOST_TEST
  uint64_t dstLevel = 3; // Evict all the way to on-chip DDR memory
  size_t numLinesMinusOne = (size * sizeof(float) - 1) >> LOG2_CACHE_LINE_BYTES;
  fence_evict_va(0, dstLevel, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmp_real)), numLinesMinusOne,
                 CACHE_LINE_BYTES, 0);
  fence_evict_va(0, dstLevel, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmp_img)), numLinesMinusOne,
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

  for (size_t index = 0; index < workBranchBits; index++) {
    barrier(numMinions, minionStep);
    if ((minionId & minionStep) == 0) {
      float* even_real = tmp_real;
      float* even_img = tmp_img;
      float* odd_real = Stack::offset(tmp_real, minionStep);
      float* odd_img = Stack::offset(tmp_img, minionStep);
      if (index == workBranchBits - 1) {
        tmp_real = result_real;
        tmp_img = result_img;
      } else {
        tmp_real = stack.push<float>(2 * size);
        tmp_img = stack.push<float>(2 * size);
      }
      twiddle_step >>= 1;
      reduce(base_twiddle_real, base_twiddle_img, twiddle_step, size, even_real, even_img, odd_real, odd_img, tmp_real,
             tmp_img);
#ifndef FFT_HOST_TEST
      fence_evict_va(0, dstLevel, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmp_real)), numLinesMinusOne,
                     CACHE_LINE_BYTES, 0);
      fence_evict_va(0, dstLevel, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tmp_img)), numLinesMinusOne,
                     CACHE_LINE_BYTES, 0);
#endif
    } else {
      break;
    }
    minionStep <<= 1;
    size <<= 1;
  }

  stack.restore(saved);
}

INLINE_ATTR void fft_threaded(size_t workBranchBits, size_t minionOffset, size_t minionId, size_t size, float* real,
                              float* img, float* result_real, float* result_img) {
  Stack stack(minionId);
  auto saved = stack.current();

  // Preccompute twiddle vector for general FFT
  float* base_twiddle_real = stack.push<float>(size);
  float* base_twiddle_img = stack.push<float>(size);
  if (size >= 16)
    twiddle_vector_big(size, base_twiddle_real, base_twiddle_img);
  else
    twiddle_vector_small(size, base_twiddle_real, base_twiddle_img);

  // Precompute twiddle vector for fft16_slice
  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);

  fft_threaded_with_precompute(workBranchBits, minionOffset, minionId, stack, base_twiddle_real, base_twiddle_img,
                               fft16_twiddle_real, fft16_twiddle_img, real, img, 0, 1, size, result_real, result_img);
  stack.restore(saved);
}

INLINE_ATTR void fft2d(size_t width, size_t height, float* real, size_t real_stride, float* img, size_t img_stride,
                       float* result_real, size_t result_real_stride, float* result_img, size_t result_img_stride) {

  size_t minionId = 0;
  Stack stack(minionId);
  auto saved = stack.current();

  assert(real_stride == img_stride);
  assert(result_real_stride == result_img_stride);

  // Precompute twiddle vector for horizontal FFT
  float* horiz_base_twiddle_real = stack.push<float>(width);
  float* horiz_base_twiddle_img = stack.push<float>(width);

  if (width >= 16)
    twiddle_vector_big(width, horiz_base_twiddle_real, horiz_base_twiddle_img);
  else
    twiddle_vector_small(width, horiz_base_twiddle_real, horiz_base_twiddle_img);

  // Precompute twiddle vector for vertical FFT
  float* vert_base_twiddle_real = stack.push<float>(height);
  float* vert_base_twiddle_img = stack.push<float>(height);
  if (height >= 16)
    twiddle_vector_big(height, vert_base_twiddle_real, vert_base_twiddle_img);
  else
    twiddle_vector_small(height, vert_base_twiddle_real, vert_base_twiddle_img);

  // Precompute twiddle vector for FFT16
  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);

  // Storage for one column of intermediate results
  float* result_column_real = stack.push<float>(height);
  float* result_column_img = stack.push<float>(height);

  constexpr size_t twiddle_step = 1;

  // Per row FFT
  for (size_t row = 0; row < height; ++row) {
    fft_with_precompute(stack, horiz_base_twiddle_real, horiz_base_twiddle_img, twiddle_step, fft16_twiddle_real,
                        fft16_twiddle_img, real + row * real_stride, img + row * img_stride, 0, 1, width,
                        result_real + row * result_real_stride, result_img + row * result_img_stride);
  }

  // Per column FFT
  for (size_t col = 0; col < width; ++col) {
    fft_with_precompute(stack, vert_base_twiddle_real, vert_base_twiddle_img, twiddle_step, fft16_twiddle_real,
                        fft16_twiddle_img, result_real, result_img, col, result_real_stride, height, result_column_real,
                        result_column_img);
    for (size_t row = 0; row < height; ++row) {
      result_real[row * result_real_stride + col] = result_column_real[row];
      result_img[row * result_img_stride + col] = result_column_img[row];
    }
  }

  stack.restore(saved);
}

template <bool pass1 = true, bool pass2 = true>
INLINE_ATTR void fft2d_threaded(size_t workRowBits, size_t workRowBranchBits, size_t workColBits,
                                size_t workColBranchBits, size_t minionOffset, size_t minionId, size_t width,
                                size_t height, float* real, size_t real_stride, float* img, size_t img_stride,
                                float* result_real, size_t result_real_stride, float* result_img,
                                size_t result_img_stride) {

  assert(workRowBits + workRowBranchBits == workColBits + workColBranchBits);

  size_t numMinions = 1 << (workRowBits + workRowBranchBits);
  if (minionId >= numMinions) {
    return;
  }

  Stack stack(minionOffset + minionId);
  auto saved = stack.current();

  assert(real_stride == img_stride);
  assert(result_real_stride == result_img_stride);

  // Precompute twiddle vector for horizontal FFT
  float* horiz_base_twiddle_real = stack.push<float>(width);
  float* horiz_base_twiddle_img = stack.push<float>(width);
  if (width >= 16)
    twiddle_vector_big(width, horiz_base_twiddle_real, horiz_base_twiddle_img);
  else
    twiddle_vector_small(width, horiz_base_twiddle_real, horiz_base_twiddle_img);

  // Precompute twiddle vector for vertical FFT
  float* vert_base_twiddle_real = stack.push<float>(height);
  float* vert_base_twiddle_img = stack.push<float>(height);
  if (height >= 16)
    twiddle_vector_big(height, vert_base_twiddle_real, vert_base_twiddle_img);
  else
    twiddle_vector_small(height, vert_base_twiddle_real, vert_base_twiddle_img);

  // Precompute twiddle vector for FFT16
  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);

  if constexpr (pass1) {
    // Per row FFT
    size_t rowsGroupSize = 1 << workRowBits;
    for (size_t row0 = 0; row0 < height; row0 += rowsGroupSize) {
      size_t rowMinionGroupId = (minionId & ((1 << (workRowBits + workRowBranchBits)) - 1)) >> workRowBranchBits;
      size_t row = row0 + rowMinionGroupId;
      size_t minionOffset0 = (minionOffset + minionId) & ~((1 << workRowBranchBits) - 1);
      size_t minionId0 = minionId & ((1 << workRowBranchBits) - 1);
      // et_printf("%s(%d) [mId=%d nMins=%d mOfs0=%d mId0=%d row=%d]\n", __func__, __LINE__, minionId, numMinions,
      // minionOffset0, minionId0, row);
      fft_threaded_with_precompute(workRowBranchBits, minionOffset0, minionId0, stack, horiz_base_twiddle_real,
                                   horiz_base_twiddle_img, fft16_twiddle_real, fft16_twiddle_img,
                                   real + row * real_stride, img + row * img_stride, 0, 1, width,
                                   result_real + row * result_real_stride, result_img + row * result_img_stride);
    }
  }

  barrier(numMinions, 1);

  if constexpr (pass2) {
    // Storage for one column of intermediate results
    float* result_column_real;
    float* result_column_img;
    result_column_real = stack.push<float>(height);
    result_column_img = stack.push<float>(height);

    // Per column FFT
    size_t colsGroupSize = 1 << workColBits;
    for (size_t col0 = 0; col0 < width; col0 += colsGroupSize) {
      size_t colMinionGroupId = (minionId & ((1 << (workColBits + workColBranchBits)) - 1)) >> workColBranchBits;
      size_t col = col0 + colMinionGroupId;
      size_t minionOffset0 = (minionOffset + minionId) & ~((1 << workColBranchBits) - 1);
      size_t minionId0 = minionId & ((1 << workColBranchBits) - 1);
      // et_printf("%s(%d) [mId=%d nMins=%d mOfs0=%d mId0=%d col=%d]\n", __func__, __LINE__, minionId, numMinions,
      // minionOffset0, minionId0, col);
      fft_threaded_with_precompute(workColBranchBits, minionOffset0, minionId0, stack, vert_base_twiddle_real,
                                   vert_base_twiddle_img, fft16_twiddle_real, fft16_twiddle_img, result_real,
                                   result_img, col, result_real_stride, height, result_column_real, result_column_img);
      for (size_t row = 0; row < height; ++row) {
#ifndef FFT_HOST_TEST
        fence_evict_va(0, 3,
                       static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&result_real[row * result_real_stride + col])),
                       0, CACHE_LINE_BYTES, 0);
        fence_evict_va(0, 3,
                       static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&result_img[row * result_img_stride + col])),
                       0, CACHE_LINE_BYTES, 0);
        uint32_t value = *reinterpret_cast<uint32_t*>(&result_column_real[row]);
        atomic_store_global_32(reinterpret_cast<uint32_t*>(&result_real[row * result_real_stride + col]), value);
        value = *reinterpret_cast<uint32_t*>(&result_column_img[row]);
        atomic_store_global_32(reinterpret_cast<uint32_t*>(&result_img[row * result_img_stride + col]), value);
#else
        result_real[row * result_real_stride + col] = result_column_real[row];
        result_img[row * result_img_stride + col] = result_column_img[row];
#endif
      }
    }
  }

  stack.restore(saved);
}

INLINE_ATTR void fft_inv_with_precompute(Stack& stack, float* base_twiddle_real, float* base_twiddle_img,
                                         size_t twiddle_step, float fft16_twiddle_real[16], float fft16_twiddle_img[16],
                                         float* real, float* img, size_t start, size_t step, size_t size,
                                         float* result_real, float* result_img) {

  auto saved = stack.current();

  // Pack the input and invert imaginary component
  float* real2 = stack.push<float>(size);
  float* img2 = stack.push<float>(size);
  for (size_t i = 0; i < size; ++i) {
    real2[i] = real[start + i * step];
    img2[i] = -img[start + i * step];
  }

  fft_with_precompute(stack, base_twiddle_real, base_twiddle_img, twiddle_step, fft16_twiddle_real, fft16_twiddle_img,
                      real2, img2, 0, 1, size, result_real, result_img);

  float reciprocal = rec(size);
  float minus_reciprocal = -reciprocal;
  for (size_t i = 0; i < size; ++i) {
    result_real[i] = reciprocal * result_real[i];
    result_img[i] = minus_reciprocal * result_img[i];
  }

  stack.restore(saved);
}

INLINE_ATTR void fft_inv(size_t size, float* real, float* img, float* result_real, float* result_img) {

  size_t minionId = 0;
  Stack stack(minionId);
  auto saved = stack.current();

  float* base_twiddle_real = stack.push<float>(size);
  float* base_twiddle_img = stack.push<float>(size);
  if (size >= 16)
    twiddle_vector_big(size, base_twiddle_real, base_twiddle_img);
  else
    twiddle_vector_small(size, base_twiddle_real, base_twiddle_img);

  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);

  constexpr size_t twiddle_step = 1;
  fft_inv_with_precompute(stack, base_twiddle_real, base_twiddle_img, twiddle_step, fft16_twiddle_real,
                          fft16_twiddle_img, real, img, 0, 1, size, result_real, result_img);

  stack.restore(saved);
}

INLINE_ATTR void fft2d_inv(size_t width, size_t height, float* real, size_t real_stride, float* img, size_t img_stride,
                           float* result_real, size_t result_real_stride, float* result_img, size_t result_img_stride) {

  size_t minionId = 0;
  Stack stack(minionId);
  auto saved = stack.current();

  assert(real_stride == img_stride);
  assert(result_real_stride == result_img_stride);

  // Precompute twiddle vector for horizontal FFT
  float* horiz_base_twiddle_real = stack.push<float>(width);
  float* horiz_base_twiddle_img = stack.push<float>(width);
  if (width >= 16)
    twiddle_vector_big(width, horiz_base_twiddle_real, horiz_base_twiddle_img);
  else
    twiddle_vector_small(width, horiz_base_twiddle_real, horiz_base_twiddle_img);

  // Precompute twiddle vector for vertical FFT
  float* vert_base_twiddle_real = stack.push<float>(height);
  float* vert_base_twiddle_img = stack.push<float>(height);
  if (height >= 16)
    twiddle_vector_big(height, vert_base_twiddle_real, vert_base_twiddle_img);
  else
    twiddle_vector_small(height, vert_base_twiddle_real, vert_base_twiddle_img);

  // Precompute twiddle vector for FFT16
  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);

  // Storage for one column of intermediate results
  float* result_column_real = stack.push<float>(height);
  float* result_column_img = stack.push<float>(height);

  constexpr size_t twiddle_step = 1;

  // Per row FFT
  for (size_t row = 0; row < height; ++row) {
    fft_inv_with_precompute(stack, horiz_base_twiddle_real, horiz_base_twiddle_img, twiddle_step, fft16_twiddle_real,
                            fft16_twiddle_img, real + row * real_stride, img + row * img_stride, 0, 1, width,
                            result_real + row * result_real_stride, result_img + row * result_img_stride);
  }

  // Per column FFT
  for (size_t col = 0; col < width; ++col) {
    fft_inv_with_precompute(stack, vert_base_twiddle_real, vert_base_twiddle_img, twiddle_step, fft16_twiddle_real,
                            fft16_twiddle_img, result_real, result_img, col, result_real_stride, height,
                            result_column_real, result_column_img);
    for (size_t row = 0; row < height; ++row) {
      result_real[row * result_real_stride + col] = result_column_real[row];
      result_img[row * result_img_stride + col] = result_column_img[row];
    }
  }

  stack.restore(saved);
}

template <bool pass1 = true, bool pass2 = true>
INLINE_ATTR void fft2d_inv_threaded([[maybe_unused]] size_t workRowBits, [[maybe_unused]] size_t workRowBranchBits,
                                    [[maybe_unused]] size_t workColBits, [[maybe_unused]] size_t workColBranchBits,
                                    [[maybe_unused]] size_t firstMinionId, size_t minionId, size_t width, size_t height,
                                    float* real, size_t real_stride, float* img, size_t img_stride, float* result_real,
                                    size_t result_real_stride, float* result_img, size_t result_img_stride) {
  Stack stack(minionId);
  auto saved = stack.current();

  assert(real_stride == img_stride);
  assert(result_real_stride == result_img_stride);

  // Precompute twiddle vector for horizontal FFT
  float* horiz_base_twiddle_real = stack.push<float>(width);
  float* horiz_base_twiddle_img = stack.push<float>(width);
  if (width >= 16)
    twiddle_vector_big(width, horiz_base_twiddle_real, horiz_base_twiddle_img);
  else
    twiddle_vector_small(width, horiz_base_twiddle_real, horiz_base_twiddle_img);

  // Precompute twiddle vector for vertical FFT
  float* vert_base_twiddle_real = stack.push<float>(height);
  float* vert_base_twiddle_img = stack.push<float>(height);
  if (height >= 16)
    twiddle_vector_big(height, vert_base_twiddle_real, vert_base_twiddle_img);
  else
    twiddle_vector_small(height, vert_base_twiddle_real, vert_base_twiddle_img);

  // Precompute twiddle vector for FFT16
  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);

  // Storage for one column of intermediate results
  float* result_column_real = stack.push<float>(height);
  float* result_column_img = stack.push<float>(height);

  constexpr size_t twiddle_step = 1;

  // Per row FFT
  for (size_t row = 0; row < height; ++row) {
    fft_inv_with_precompute(stack, horiz_base_twiddle_real, horiz_base_twiddle_img, twiddle_step, fft16_twiddle_real,
                            fft16_twiddle_img, real + row * real_stride, img + row * img_stride, 0, 1, width,
                            result_real + row * result_real_stride, result_img + row * result_img_stride);
  }

  // Per column FFT
  for (size_t col = 0; col < width; ++col) {
    fft_inv_with_precompute(stack, vert_base_twiddle_real, vert_base_twiddle_img, twiddle_step, fft16_twiddle_real,
                            fft16_twiddle_img, result_real, result_img, col, result_real_stride, height,
                            result_column_real, result_column_img);
    for (size_t row = 0; row < height; ++row) {
      result_real[row * result_real_stride + col] = result_column_real[row];
      result_img[row * result_img_stride + col] = result_column_img[row];
    }
  }

  stack.restore(saved);
}

} // namespace dnn_lib

#endif // _FFT_H_

#ifdef FFT_HOST_TEST

// Build and run a test on the host as:
//
//   cp FFT.h /tmp/fft.cpp && g++ -DFFT_HOST_TEST=1 /tmp/fft.cpp -g && ./a.out

#include <iostream>

void print(size_t height, size_t width, const std::string& name, float* real, size_t real_stride, float* img,
           size_t img_stride) {
  std::cout << "\n" << name << "\n";
  for (size_t i = 0; i < height; ++i) {
    for (size_t j = 0; j < width; ++j) {
      if (j != 0) {
        std::cout << " ; ";
      }
      std::cout << real[i * real_stride + j] << "," << img[i * img_stride + j];
    }
    std::cout << "\n";
  }
}

void test1() {
  std::cout << "\n>>>> Test 1\n";

  size_t size = 32;
  float result_real[size];

  float result_img[size];
  float real[size];
  float img[size];
  for (size_t i = 0; i < size; ++i) {
    real[i] = i;
    img[i] = 0;
  }
  print(1, size, "Input", real, 0, img, 0);

  dnn_lib::fft(size, real, img, result_real, result_img);
  print(1, size, "FFT", result_real, 0, result_img, 0);

  float expected_result_real[size] = {496., -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.,
                                      -16., -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.,
                                      -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.};
  float expected_result_img[size] = {
    +0.,  +162.4507262, +80.43743187, +52.74493134, +38.627417, +29.93389459, +23.9456922,  +19.49605641,
    +16., +13.13086065, +10.69085821, +8.55217818,  +6.627417,  +4.85354694,  +3.18259788,  +1.57586245,
    +0.,  -1.57586245,  -3.18259788,  -4.85354694,  -6.627417,  -8.55217818,  -10.69085821, -13.13086065,
    -16., -19.49605641, -23.9456922,  -29.93389459, -38.627417, -52.74493134, -80.43743187, -162.4507262};
  for (size_t i = 0; i < size; ++i) {
    assert(std::abs(result_real[i] - expected_result_real[i]) < 0.01f);
    assert(std::abs(result_img[i] - expected_result_img[i]) < 0.01f);
  }

  float reconst_real[size];
  float reconst_img[size];
  dnn_lib::fft_inv(size, result_real, result_img, reconst_real, reconst_img);
  print(1, size, "Reconstructed", reconst_real, 0, reconst_img, 0);

  for (size_t i = 0; i < size; ++i) {
    assert(std::abs(reconst_real[i] - real[i]) < 0.001f);
    assert(std::abs(reconst_img[i] - img[i]) < 0.001f);
  }
}

void test2() {

  std::cout << "\n>>>> Test 2\n";

  constexpr size_t height = 2;
  constexpr size_t width = 4;

  constexpr size_t real_stride = 4;
  constexpr size_t img_stride = 4;
  float real[height][real_stride] = {{1, 3, 2, 1}, {1, 2, 2, 0}};
  float img[height][img_stride] = {0};
  print(height, width, "Input", &real[0][0], real_stride, &img[0][0], img_stride);

  constexpr size_t result_real_stride = 4;
  constexpr size_t result_img_stride = 4;
  float result_real[height][result_real_stride] = {0};
  float result_img[height][img_stride] = {0};
  dnn_lib::fft2d(width, height, &real[0][0], real_stride, &img[0][0], img_stride, &result_real[0][0],
                 result_real_stride, &result_img[0][0], result_img_stride);
  print(height, width, "FFT 2D", &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);

  float expected_result_real[height][result_real_stride] = {{12., -2., 0., -2.}, {2., 0., -2., 0.}};
  float expected_result_img[height][img_stride] = {{0., -4., 0., 4.}, {0., 0., 0., 0.}};
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      assert(std::abs(result_real[row][col] - expected_result_real[row][col]) < 0.00001f);
      assert(std::abs(result_img[row][col] - expected_result_img[row][col]) < 0.00001f);
    }
  }

  std::cout << "Reconstructed\n";

  constexpr size_t reconst_real_stride = 4;
  constexpr size_t reconst_img_stride = 4;
  float reconst_real[height][reconst_real_stride] = {0};
  float reconst_img[height][reconst_img_stride] = {0};

  dnn_lib::fft2d_inv(width, height, &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride,
                     &reconst_real[0][0], reconst_real_stride, &reconst_img[0][0], reconst_img_stride);
  print(height, width, "Reconstructed", &reconst_real[0][0], reconst_real_stride, &reconst_img[0][0],
        reconst_img_stride);

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      assert(std::abs(reconst_real[row][col] - real[row][col]) < 0.000001f);
      assert(std::abs(reconst_img[row][col] - img[row][col]) < 0.000001f);
    }
  }
}

void test3() {

  std::cout << "\n>>>> Test 3\n";

  constexpr size_t height = 1;
  constexpr size_t width = 1;

  constexpr size_t real_stride = width;
  constexpr size_t img_stride = width;

  float real[height][real_stride];
  float img[height][img_stride];
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      real[row][col] = row;
      img[row][col] = col;
    }
  }
  print(height, width, "Input", &real[0][0], real_stride, &img[0][0], img_stride);

  constexpr size_t result_real_stride = width;
  constexpr size_t result_img_stride = width;
  float result_real[height][result_real_stride] = {0};
  float result_img[height][img_stride] = {0};
  dnn_lib::fft2d(width, height, &real[0][0], real_stride, &img[0][0], img_stride, &result_real[0][0],
                 result_real_stride, &result_img[0][0], result_img_stride);
  print(height, width, "FFT 2D", &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);
}

void test4() {
  std::cout << "\n>>>> Test 4\n";

  size_t size = 32;
  float result_real[size];

  float result_img[size];
  float real[size];
  float img[size];
  for (size_t i = 0; i < size; ++i) {
    real[i] = i;
    img[i] = 0;
  }
  print(1, size, "Input", real, 0, img, 0);

  constexpr size_t minionOffset = 16;
  constexpr size_t workBranchBits = 2;
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
  dnn_lib::fft_threaded(workBranchBits, minionOffset, 1, size, real, img, result_real, result_img);
  dnn_lib::fft_threaded(workBranchBits, minionOffset, 3, size, real, img, result_real, result_img);
  dnn_lib::fft_threaded(workBranchBits, minionOffset, 2, size, real, img, result_real, result_img);
  dnn_lib::fft_threaded(workBranchBits, minionOffset, 0, size, real, img, result_real, result_img);

  float expected_result_real[size] = {496., -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.,
                                      -16., -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.,
                                      -16., -16., -16., -16., -16., -16., -16., -16., -16., -16.};
  float expected_result_img[size] = {
    +0.,  +162.4507262, +80.43743187, +52.74493134, +38.627417, +29.93389459, +23.9456922,  +19.49605641,
    +16., +13.13086065, +10.69085821, +8.55217818,  +6.627417,  +4.85354694,  +3.18259788,  +1.57586245,
    +0.,  -1.57586245,  -3.18259788,  -4.85354694,  -6.627417,  -8.55217818,  -10.69085821, -13.13086065,
    -16., -19.49605641, -23.9456922,  -29.93389459, -38.627417, -52.74493134, -80.43743187, -162.4507262};
  for (size_t i = 0; i < size; ++i) {
    assert(std::abs(result_real[i] - expected_result_real[i]) < 0.01f);
    assert(std::abs(result_img[i] - expected_result_img[i]) < 0.01f);
  }

  float reconst_real[size];
  float reconst_img[size];
  dnn_lib::fft_inv(size, result_real, result_img, reconst_real, reconst_img);
  print(1, size, "Reconstructed", reconst_real, 0, reconst_img, 0);

  for (size_t i = 0; i < size; ++i) {
    assert(std::abs(reconst_real[i] - real[i]) < 0.001f);
    assert(std::abs(reconst_img[i] - img[i]) < 0.001f);
  }
}

void test5() {

  std::cout << "\n>>>> Test 5\n";

  constexpr size_t height = 2;
  constexpr size_t width = 4;

  constexpr size_t real_stride = 4;
  constexpr size_t img_stride = 4;
  float real[height][real_stride] = {{1, 3, 2, 1}, {1, 2, 2, 0}};
  float img[height][img_stride] = {0};
  print(height, width, "Input", &real[0][0], real_stride, &img[0][0], img_stride);

  size_t workRowBits = 1;
  size_t workRowBranchBits = 1;
  size_t workColBits = 2;
  size_t workColBranchBits = 0;
  constexpr size_t minionOffset = 16;
  constexpr size_t result_real_stride = 4;
  constexpr size_t result_img_stride = 4;
  float result_real[height][result_real_stride] = {0};
  float result_img[height][img_stride] = {0};
  dnn_lib::fft2d_threaded<true, false>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset, 1,
                                       width, height, &real[0][0], real_stride, &img[0][0], img_stride,
                                       &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);
  dnn_lib::fft2d_threaded<true, false>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset, 0,
                                       width, height, &real[0][0], real_stride, &img[0][0], img_stride,
                                       &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);
  dnn_lib::fft2d_threaded<true, false>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset, 3,
                                       width, height, &real[0][0], real_stride, &img[0][0], img_stride,
                                       &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);
  dnn_lib::fft2d_threaded<true, false>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset, 2,
                                       width, height, &real[0][0], real_stride, &img[0][0], img_stride,
                                       &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);
  dnn_lib::fft2d_threaded<false, true>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset, 0,
                                       width, height, &real[0][0], real_stride, &img[0][0], img_stride,
                                       &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);
  dnn_lib::fft2d_threaded<false, true>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset, 1,
                                       width, height, &real[0][0], real_stride, &img[0][0], img_stride,
                                       &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);
  dnn_lib::fft2d_threaded<false, true>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset, 2,
                                       width, height, &real[0][0], real_stride, &img[0][0], img_stride,
                                       &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);
  dnn_lib::fft2d_threaded<false, true>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset, 3,
                                       width, height, &real[0][0], real_stride, &img[0][0], img_stride,
                                       &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);
  print(height, width, "FFT 2D", &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride);

  float expected_result_real[height][result_real_stride] = {{12., -2., 0., -2.}, {2., 0., -2., 0.}};
  float expected_result_img[height][img_stride] = {{0., -4., 0., 4.}, {0., 0., 0., 0.}};
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      assert(std::abs(result_real[row][col] - expected_result_real[row][col]) < 0.00001f);
      assert(std::abs(result_img[row][col] - expected_result_img[row][col]) < 0.00001f);
    }
  }

  std::cout << "Reconstructed\n";

  constexpr size_t reconst_real_stride = 4;
  constexpr size_t reconst_img_stride = 4;
  float reconst_real[height][reconst_real_stride] = {0};
  float reconst_img[height][reconst_img_stride] = {0};

  dnn_lib::fft2d_inv(width, height, &result_real[0][0], result_real_stride, &result_img[0][0], result_img_stride,
                     &reconst_real[0][0], reconst_real_stride, &reconst_img[0][0], reconst_img_stride);
  print(height, width, "Reconstructed", &reconst_real[0][0], reconst_real_stride, &reconst_img[0][0],
        reconst_img_stride);

  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      assert(std::abs(reconst_real[row][col] - real[row][col]) < 0.000001f);
      assert(std::abs(reconst_img[row][col] - img[row][col]) < 0.000001f);
    }
  }
}

int main(int argc, char** argv) {

  test1();
  test2();
  test3();
  test4();
  test5();
  std::cout << "\nALL PASSED\n\n";

  return 0;
}

#endif
