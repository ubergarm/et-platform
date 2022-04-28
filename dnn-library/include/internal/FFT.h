#ifndef _FFT_H_
#define _FFT_H_

#include <cassert>
#include <cmath>
#include <cstddef>

namespace dnn_lib {

// Prototypes

INLINE_ATTR void fft(size_t size, float *real, float *img, float *result_real,
         float *result_img);

INLINE_ATTR void fft2d(size_t width, size_t height, float *real, size_t real_stride,
           float *img, size_t img_stride, float *result_real,
           size_t result_real_stride, float *result_img,
           size_t result_img_stride);

INLINE_ATTR void fft_inv(size_t size, float *real, float *img, float *result_real,
             float *result_img);  

INLINE_ATTR void fft2d_inv(size_t width, size_t height, float *real, size_t real_stride,
               float *img, size_t img_stride, float *result_real,
               size_t result_real_stride, float *result_img,
               size_t result_img_stride);

// Implementation

INLINE_ATTR void euler_formula(float angle, float &real, float &img) {
  real = cos(angle);
  img = sin(angle);
}

INLINE_ATTR void w(size_t j, size_t n, float &real, float &img) {
  return euler_formula(static_cast<float>(-2 * M_PI) * static_cast<float>(j) / float(n), real, img);
}

INLINE_ATTR void twiddle_vector_small(size_t size, float real[], float img[]) {  
  for (size_t i = 0; i < size; ++i) {
    w(i, size, real[i], img[i]);
  }
}

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

constexpr bool isPowerOfTwo(size_t value) { return countBits(value) == 1; }

static_assert(isPowerOfTwo(0) == false);
static_assert(isPowerOfTwo(8) == true);
static_assert(isPowerOfTwo(9) == false);

INLINE_ATTR void twiddle_vector_big(size_t n, float real[], float img[]) {
  assert(n >= 16 and isPowerOfTwo(n));
  real[0] = 1;
  img[0] = 0;
  real[n >> 2] = 0;
  img[n >> 2] = -1;
  real[n >> 1] = -1;
  img[n >> 1] = 0;
  const float k = static_cast<float>(2 * M_PI) / static_cast<float>(n);
  for (size_t j = 1; j < n / 8; ++j) {
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
  const float sine = sin(static_cast<float>(M_PI * 0.25));
  real[n >> 3] = sine;
  img[n >> 3] = -sine;
  real[(n >> 3) + (n >> 2)] = -sine;
  img[(n >> 3) + (n >> 2)] = -sine;
  real[(n >> 3) + (n >> 1)] = -sine;
  img[(n >> 3) + (n >> 1)] = sine;
  real[n - (n >> 3)] = sine;
  img[n - (n >> 3)] = sine;
}

INLINE_ATTR void mult(float x, float y, float u, float v, float &real, float &img) {
  real = x * u - y * v;
  img = x * v + y * u;
}

INLINE_ATTR void add(float x, float y, float u, float v, float &real, float &img) {
  real = x + u;
  img = y + v;
}

INLINE_ATTR void sub(float x, float y, float u, float v, float &real, float &img) {
  real = x - u;
  img = y - v;
}

constexpr size_t twiddle_index(size_t round, size_t i) {
  return i & ((8 + 16 + 32 + 64 + 128) >> round);
}

INLINE_ATTR void fft16_round(float twiddle_real[16], float twiddle_img[16],
                        float X_real[16], float X_img[16], size_t round,
                        float result_real[16], float result_img[16],
                        const size_t select_mult_second[8],
                        const size_t select_add_or_sub_first[8]) {

  // This loop is suitable for vectorizing
  for (size_t i = 0; i < 8; ++i) {
    size_t tidx = twiddle_index(round, i);
    size_t sidx = select_mult_second[i];
    size_t fidx = select_add_or_sub_first[i];
    float term_real, term_img;
    mult(twiddle_real[tidx], twiddle_img[tidx], X_real[sidx], X_img[sidx],
         term_real, term_img);
    add(X_real[fidx], X_img[fidx], term_real, term_img, result_real[i],
        result_img[i]);
    sub(X_real[fidx], X_img[fidx], term_real, term_img, result_real[i + 8],
        result_img[i + 8]);
  }
}

INLINE_ATTR void fft16_slice(float *real, float *img, size_t start, size_t step,
                        size_t size, float twiddle_real[16],
                        float twiddle_img[16], float res_real[16],
                        float res_img[16]) {
  assert(size == 16);
  float tmp_real[16];
  float tmp_img[16];

  size_t select_mult_second[8] = {8, 12, 10, 14, 9, 13, 11, 15};
  size_t select_add_or_sub_first[8] = {0, 4, 2, 6, 1, 5, 3, 7};

  // This loop is suitable for vectorizing
  for (size_t i = 0; i < 8; ++i) {
    select_mult_second[i] = start + step * select_mult_second[i];
    select_add_or_sub_first[i] = start + step * select_add_or_sub_first[i];
  }

  fft16_round(twiddle_real, twiddle_img, real, img, 0, tmp_real, tmp_img,
              select_mult_second, select_add_or_sub_first);

  constexpr size_t select_mult_second2[8] = {1, 3, 5, 7, 9, 11, 13, 15};
  constexpr size_t select_add_or_sub_first2[8] = {0, 2, 4, 6, 8, 10, 12, 14};

  fft16_round(twiddle_real, twiddle_img, tmp_real, tmp_img, 1, res_real,
              res_img, select_mult_second2, select_add_or_sub_first2);

  fft16_round(twiddle_real, twiddle_img, res_real, res_img, 2, tmp_real,
              tmp_img, select_mult_second2, select_add_or_sub_first2);

  fft16_round(twiddle_real, twiddle_img, tmp_real, tmp_img, 3, res_real,
              res_img, select_mult_second2, select_add_or_sub_first2);
}

void fft_with_precompute(float *base_twiddle_real, float *base_twiddle_img,
                         size_t twiddle_step, float fft16_twiddle_real[16],
                         float fft16_twiddle_img[16], float *real, float *img,
                         size_t start, size_t step, size_t size,
                         float *result_real, float *result_img) {
  if (size == 16) {
    fft16_slice(real, img, start, step, size, fft16_twiddle_real,
                fft16_twiddle_img, result_real, result_img);
  } else if (size == 1) {
    result_real[0] = real[start];
    result_img[0] = img[start];
  } else {
    size_t half_size = size >> 1;    
    float tmp_real_even[half_size];
    float tmp_img_even[half_size];
    fft_with_precompute(base_twiddle_real, base_twiddle_img, 2 * twiddle_step,
                        fft16_twiddle_real, fft16_twiddle_img, real, img, start,
                        2 * step, half_size, tmp_real_even, tmp_img_even);
    float tmp_real_odd[half_size];
    float tmp_img_odd[half_size];
    fft_with_precompute(base_twiddle_real, base_twiddle_img, 2 * twiddle_step,
                        fft16_twiddle_real, fft16_twiddle_img, real, img,
                        start + step, 2 * step, half_size, tmp_real_odd,
                        tmp_img_odd);
    size_t twiddle_index = 0;    
    for (size_t j = 0; j < half_size; ++j) {
      float twiddle_real = base_twiddle_real[twiddle_index];
      float twiddle_img = base_twiddle_img[twiddle_index];
      float term_real, term_img;
      mult(twiddle_real, twiddle_img, tmp_real_odd[j], tmp_img_odd[j],
           term_real, term_img);
      add(tmp_real_even[j], tmp_img_even[j], term_real, term_img,
          result_real[j], result_img[j]);
      sub(tmp_real_even[j], tmp_img_even[j], term_real, term_img,
          result_real[j + (half_size)], result_img[j + (half_size)]);
      twiddle_index += twiddle_step;
    }
  }
}

INLINE_ATTR void fft(size_t size, float *real, float *img, float *result_real,
         float *result_img) {
  float base_twiddle_real[size];
  float base_twiddle_img[size];
  if (size >= 16)
    twiddle_vector_big(size, base_twiddle_real, base_twiddle_img);
  else
    twiddle_vector_small(size, base_twiddle_real, base_twiddle_img);
  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);
  constexpr size_t twiddle_step = 1;
  fft_with_precompute(base_twiddle_real, base_twiddle_img, twiddle_step,
                      fft16_twiddle_real, fft16_twiddle_img, real, img, 0, 1,
                      size, result_real, result_img);
}

INLINE_ATTR void fft2d(size_t width, size_t height, float *real, size_t real_stride,
           float *img, size_t img_stride, float *result_real,
           size_t result_real_stride, float *result_img,
           size_t result_img_stride) {

  assert(real_stride == img_stride);
  assert(result_real_stride == result_img_stride);

  // Precompute twiddle vector for horizontal FFT
  float horiz_base_twiddle_real[width];
  float horiz_base_twiddle_img[width];
  if (width >= 16)
    twiddle_vector_big(width, horiz_base_twiddle_real, horiz_base_twiddle_img);
  else
    twiddle_vector_small(width, horiz_base_twiddle_real,
                         horiz_base_twiddle_img);

  // Precompute twiddle vector for vertical FFT
  float vert_base_twiddle_real[width];
  float vert_base_twiddle_img[width];
  if (height >= 16)
    twiddle_vector_big(height, vert_base_twiddle_real, vert_base_twiddle_img);
  else
    twiddle_vector_small(height, vert_base_twiddle_real, vert_base_twiddle_img);

  // Precompute twiddle vector for FFT16
  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);

  // Storage for one column of intermediate results
  float result_column_real[height];
  float result_column_img[height];

  constexpr size_t twiddle_step = 1;

  // Per row FFT
  for (size_t row = 0; row < height; ++row) {
    fft_with_precompute(horiz_base_twiddle_real, horiz_base_twiddle_img,
                        twiddle_step, fft16_twiddle_real, fft16_twiddle_img,
                        real + row * real_stride, img + row * img_stride, 0, 1,
                        width, result_real + row * result_real_stride,
                        result_img + row * result_img_stride);
  }

  // Per column FFT
  for (size_t col = 0; col < width; ++col) {
    fft_with_precompute(vert_base_twiddle_real, vert_base_twiddle_img,
                        twiddle_step, fft16_twiddle_real, fft16_twiddle_img,
                        result_real, result_img, col, result_real_stride,
                        height, result_column_real, result_column_img);
    for (size_t row = 0; row < height; ++row) {
      result_real[row * result_real_stride + col] = result_column_real[row];
      result_img[row * result_img_stride + col] = result_column_img[row];
    }
  }
}

INLINE_ATTR void fft_inv_with_precompute(float *base_twiddle_real, float *base_twiddle_img,
                             size_t twiddle_step, float fft16_twiddle_real[16],
                             float fft16_twiddle_img[16], float *real,
                             float *img, size_t start, size_t step, size_t size,
                             float *result_real, float *result_img) {

  float real2[size];
  float img2[size];
  for (size_t i = 0; i < size; ++i) {
    real2[i] = real[start + i * step];
    img2[i] = -img[start + i * step];
  }

  fft_with_precompute(base_twiddle_real, base_twiddle_img, twiddle_step,
                      fft16_twiddle_real, fft16_twiddle_img, real2, img2, 0, 1,
                      size, result_real, result_img);

  float rec = 1.f / static_cast<float>(size);
  float minus_rec = -rec;

  for (size_t i = 0; i < size; ++i) {
    result_real[i] = rec * result_real[i];
    result_img[i] = minus_rec * result_img[i];
  }
}

INLINE_ATTR void fft_inv(size_t size, float *real, float *img, float *result_real,
             float *result_img) {
  float base_twiddle_real[size];
  float base_twiddle_img[size];
  if (size >= 16)
    twiddle_vector_big(size, base_twiddle_real, base_twiddle_img);
  else
    twiddle_vector_small(size, base_twiddle_real, base_twiddle_img);
  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);
  constexpr size_t twiddle_step = 1;
  fft_inv_with_precompute(base_twiddle_real, base_twiddle_img, twiddle_step,
                          fft16_twiddle_real, fft16_twiddle_img, real, img, 0,
                          1, size, result_real, result_img);
}

INLINE_ATTR void fft2d_inv(size_t width, size_t height, float *real, size_t real_stride,
               float *img, size_t img_stride, float *result_real,
               size_t result_real_stride, float *result_img,
               size_t result_img_stride) {

  assert(real_stride == img_stride);
  assert(result_real_stride == result_img_stride);

  // Precompute twiddle vector for horizontal FFT
  float horiz_base_twiddle_real[width];
  float horiz_base_twiddle_img[width];
  if (width >= 16)
    twiddle_vector_big(width, horiz_base_twiddle_real, horiz_base_twiddle_img);
  else
    twiddle_vector_small(width, horiz_base_twiddle_real,
                         horiz_base_twiddle_img);

  // Precompute twiddle vector for vertical FFT
  float vert_base_twiddle_real[width];
  float vert_base_twiddle_img[width];
  if (height >= 16)
    twiddle_vector_big(height, vert_base_twiddle_real, vert_base_twiddle_img);
  else
    twiddle_vector_small(height, vert_base_twiddle_real, vert_base_twiddle_img);

  // Precompute twiddle vector for FFT16
  float fft16_twiddle_real[16];
  float fft16_twiddle_img[16];
  twiddle_vector_big(16, fft16_twiddle_real, fft16_twiddle_img);

  // Storage for one column of intermediate results
  float result_column_real[height];
  float result_column_img[height];

  constexpr size_t twiddle_step = 1;

  // Per row FFT
  for (size_t row = 0; row < height; ++row) {
    fft_inv_with_precompute(horiz_base_twiddle_real, horiz_base_twiddle_img,
                            twiddle_step, fft16_twiddle_real, fft16_twiddle_img,
                            real + row * real_stride, img + row * img_stride, 0,
                            1, width, result_real + row * result_real_stride,
                            result_img + row * result_img_stride);
  }

  // Per column FFT
  for (size_t col = 0; col < width; ++col) {
    fft_inv_with_precompute(vert_base_twiddle_real, vert_base_twiddle_img,
                            twiddle_step, fft16_twiddle_real, fft16_twiddle_img,
                            result_real, result_img, col, result_real_stride,
                            height, result_column_real, result_column_img);
    for (size_t row = 0; row < height; ++row) {
      result_real[row * result_real_stride + col] = result_column_real[row];
      result_img[row * result_img_stride + col] = result_column_img[row];
    }
  }
}

} // namespace dnn_lib

#endif // _FFT_H_