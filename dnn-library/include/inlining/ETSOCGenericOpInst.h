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

#ifndef _ETSOCGENERICOP_INST_H_
#define _ETSOCGENERICOP_INST_H_

#include "DenoiseMasks.h"
#include "FFT.h"
#include "LibTensor.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

enum class Operation {
  FFT = 0,
  IFFT,
  NOISE_FILTER_1,
  COUNT,
};

static constexpr const char* Op2String[] = {"FFT", "IFFT", "NOISE_FILTER_1"};

template <typename T> INLINE_ATTR T min(T a, T b) {
  return (a < b) ? a : b;
}

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

void fftTiling(size_t batches, [[maybe_unused]] size_t channels, [[maybe_unused]] size_t components, size_t height,
               size_t width, size_t numMinions, size_t& workBatchBits, size_t& workRowBits, size_t& workRowBranchBits,
               size_t& workColBits, size_t& workColBranchBits) {

  size_t log2NumMinions = log2(numMinions);

  workRowBits = min(size_t(5), min(log2(height), log2(width)));
  workRowBranchBits = 0;
  workColBits = workRowBits;
  workColBranchBits = workRowBranchBits;
  workBatchBits = min(log2(batches), log2NumMinions - workRowBits - workRowBranchBits);

  assert(workRowBits + workRowBranchBits == workColBits + workColBranchBits);
  assert(workRowBits + workRowBranchBits <= 5);
}

template <bool inverse = false>
INLINE_ATTR void fft(LibTensor* outT, LibTensor* inT, [[maybe_unused]] uint64_t flags, const uint32_t minionOffset,
                     uint32_t numMinions, uint32_t minionId) {

  const dim_t* srcDims = inT->dims().data();
  const dim_t* dstDims = outT->dims().data();
  const dim_t* srcStrides = inT->strides().data();
  const dim_t* dstStrides = outT->strides().data();

  size_t batches = srcDims[0];
  size_t channels = srcDims[1];
  [[maybe_unused]] size_t components = srcDims[2];
  size_t height = srcDims[3];
  size_t width = srcDims[4];

  assert(batches > 0);
  assert(channels > 0);
  assert(components == 2);
  assert(height > 0);
  assert(width > 0);

  assert(isPowerOfTwo(height));
  assert(isPowerOfTwo(width));

  assert(batches == dstDims[0]);
  assert(channels == dstDims[1]);
  assert(components == dstDims[2]);
  assert(height == dstDims[3]);
  assert(width == dstDims[4]);

  // Mapping from minion dimensions to compute dimensions
  size_t workBatchBits;
  size_t workRowBits, workRowBranchBits;
  size_t workColBits, workColBranchBits;
  fftTiling(batches, channels, components, height, width, numMinions, workBatchBits, workRowBits, workRowBranchBits,
            workColBits, workColBranchBits);

  if (minionId - minionOffset == 0) {
    et_printf("bBits=%d rBits=%d rBrBits=%d cBits=%d cBrBits=%d", workBatchBits, workRowBits, workRowBranchBits,
              workColBits, workColBranchBits);
  }

  // Ensure we got assigned at least as many minions as we can use
  assert(numMinions >= static_cast<size_t>(1 << (workBatchBits + workRowBits + workRowBranchBits)));

  // Use just as many minions as we can
  numMinions = min(numMinions, static_cast<uint32_t>(1 << (workBatchBits + workRowBits + workRowBranchBits)));

  // Unused minions return inmediately
  if ((minionId - minionOffset) >= numMinions) {
    return;
  }

  // et_printf("%s(%d) [minionOffset=%d numMinions=%d minionId=%d]\n", __func__, __LINE__, minionOffset, numMinions,
  // minionId);

  float* in = inT->getRawDataPointer<float>();
  float* out = outT->getRawDataPointer<float>();

  size_t batchElemsGroupSize = 1 << workBatchBits;

  for (size_t batch0 = 0; batch0 < batches; batch0 += batchElemsGroupSize) {

    size_t batchMinionGroupId =
      (minionId & ((1 << (workBatchBits + workColBits + workColBranchBits)) - 1)) >> (workColBits + workColBranchBits);
    size_t batch = batch0 + batchMinionGroupId;
    size_t minionOffset0 = (minionOffset + minionId) & ~((1 << (workColBits + workColBranchBits)) - 1);
    size_t minionId0 = minionId & ((1 << (workColBits + workColBranchBits)) - 1);

    et_printf("%s(%d) [mId=%d nMins=%d mOfs0=%d mId0=%d batch=%d]\n", __func__, __LINE__, minionId, numMinions,
              minionOffset0, minionId0, batch);

    for (size_t channel = 0; channel < channels; ++channel) {

      float* real = in + srcStrides[0] * batch + srcStrides[1] * channel;
      size_t real_stride = srcStrides[3];
      float* img = real + srcStrides[2];
      size_t img_stride = srcStrides[3];
      float* result_real = out + dstStrides[0] * batch + dstStrides[1] * channel;
      size_t result_real_stride = dstStrides[3];
      float* result_img = result_real + dstStrides[2];
      size_t result_img_stride = dstStrides[3];

      if constexpr (inverse) {
        constexpr bool pass1 = true;
        constexpr bool pass2 = true;
        fft2d_inv_threaded<pass1, pass2>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset0,
                                         minionId0, width, height, real, real_stride, img, img_stride, result_real,
                                         result_real_stride, result_img, result_img_stride);
      } else {
        fft2d_threaded(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset0, minionId0, width,
                       height, real, real_stride, img, img_stride, result_real, result_real_stride, result_img,
                       result_img_stride);
      }
    }
  }
}

INLINE_ATTR void freqDomainNoiseFilter(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset,
                                       const uint32_t numMinions, uint32_t minionId) {
  //  FIXME: just minon 0 does some work at the moment.
  if ((minionId - minionOffset) != 0) {
    return;
  }

  et_printf("%s(%d) [%d]\n", __func__, __LINE__, minionId);

  (void)flags;
  (void)minionOffset;
  (void)numMinions;

  et_assert(inT->dims()[0] == 2);

  auto w = inT->dims()[1];
  auto h = inT->dims()[2];
  auto inH = inT->getHandle<float>();
  auto outH = outT->getHandle<float>();
  // elementwise product of real and imaginary planes with the mask.
  // it is just  a PoC. it can probably be just fused with fft.
  // (Evey real or imaginary output emited or not based on mask
  // also SIMD mask's can be used ot skip when fft is vectorized.
  for (dim_t n = 0; n < w; n++) {
    for (dim_t m = 0; m < h; m++) {
      std::array<dim_t, 3> posReal = {0, m, n};
      std::array<dim_t, 3> posImg = {1, m, n};
      auto mask = denoiseMask[m * w + n];

      outH.at(posReal) = inH.at(posReal) * mask;
      outH.at(posImg) = inH.at(posImg) * mask;
    }
  }
}

template <ElemKind elK>
INLINE_ATTR void fwdLibETSOCGenericOpInst(LibTensor* outT, LibTensor* inT, uint32_t op, uint64_t flags,
                                          const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  static_assert(elK == FloatTy);

  (void)flags;

  // If assigned minions is 0, use them all
  size_t numMinions = (assignedMinions == 0) ? static_cast<uint32_t>(MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  size_t minionId = get_minion_id();

  if (minionId == 0) {
    et_printf("%s(%d) [numMinions=%d op=%s]\n", __func__, __LINE__, numMinions, Op2String[op]);
  }

  switch (Operation(op)) {
  case Operation::FFT:
    return fft<false>(outT, inT, flags, minionOffset, numMinions, minionId);
  case Operation::IFFT:
    return fft<true>(outT, inT, flags, minionOffset, numMinions, minionId);
  case Operation::NOISE_FILTER_1:
    return freqDomainNoiseFilter(outT, inT, flags, minionOffset, numMinions, minionId);
  case Operation::COUNT:
  default:
    et_assert("unsupported operation");
  }

  return;
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ETSOCGENERICOP_INST_H_
