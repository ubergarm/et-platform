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

INLINE_ATTR void fftTiling(size_t batches, [[maybe_unused]] size_t channels, [[maybe_unused]] size_t components,
                           size_t height, size_t width, size_t numMinions, size_t& workBatchBits, size_t& workRowBits,
                           size_t& workRowBranchBits, size_t& workColBits, size_t& workColBranchBits) {

  // About to assign all the bits
  size_t availableBits = log2(numMinions);

  // Assign as many bits as possible in the batch element dimension
  workBatchBits = min(log2(batches), availableBits);
  availableBits -= workBatchBits;

  if (height >= width) {
    // Assign as many bits as possible in the rows dimension
    workRowBits = min(log2(height), availableBits);
    availableBits -= workRowBits;

    // Assign the remaiming bits on the row branches dimension
    workRowBranchBits = min(log2(width), availableBits);

    // Assign as many minion bits for the columns dimension as
    // done for the rows dimension. If the columns dimension is
    // not as big then do tiling also on branches.
    workColBits = min(log2(width), workRowBits);
    workColBranchBits = workRowBits + workRowBranchBits - workColBits;
  } else {
    // Assign as many bits as possible in the rows dimension
    workColBits = min(log2(width), availableBits);
    availableBits -= workColBits;

    // Assign the remaiming bits on the row branches dimension
    workColBranchBits = min(log2(height), availableBits);

    // Assign as many minion bits for the columns dimension as
    // done for the rows dimension. If the columns dimension is
    // not as big then do tiling also on branches.
    workRowBits = min(log2(height), workColBits);
    workRowBranchBits = workColBits + workColBranchBits - workRowBits;
  }

  // Overriding tiling?
  if constexpr (false) {
    workBatchBits = 0;
    workRowBits = 8;
    workRowBranchBits = 2;
    workColBits = 8;
    workColBranchBits = 2;
  }

  assert(workRowBits + workRowBranchBits == workColBits + workColBranchBits);
  assert(workBatchBits + workRowBits + workRowBranchBits <= log2(numMinions));
}

template <bool inverse = false>
INLINE_ATTR void fft(LibTensor* outT, LibTensor* inT, [[maybe_unused]] uint64_t flags, const uint32_t minionOffset,
                     size_t numMinions, size_t minionId) {

  const dim_t* srcDims = inT->dims().data();
  [[maybe_unused]] const dim_t* dstDims = outT->dims().data();
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
  numMinions = min(numMinions, 1UL << (workBatchBits + workRowBits + workRowBranchBits));

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

    // et_printf("%s(%d) [mId=%d nMins=%d mOfs0=%d mId0=%d batch=%d]\n", __func__, __LINE__, minionId, numMinions,
    //          minionOffset0, minionId0, batch);

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
        fft2d_inv_threaded<pass1, pass2>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset,
                                         minionOffset0, minionId0, width, height, real, real_stride, img, img_stride,
                                         result_real, result_real_stride, result_img, result_img_stride);
      } else {
        fft2d_threaded(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset, minionOffset0,
                       minionId0, width, height, real, real_stride, img, img_stride, result_real, result_real_stride,
                       result_img, result_img_stride);
      }
    }
  }
}

INLINE_ATTR void freqDomainNoiseFilter(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset,
                                       size_t numMinions, size_t minionId) {
  //  FIXME: just minon 0 does some work at the moment.
  if ((minionId - minionOffset) != 0) {
    return;
  }

  et_printf("%s(%d) [%d]\n", __func__, __LINE__, minionId);

  (void)flags;
  (void)minionOffset;
  (void)numMinions;

  auto images = inT->dims()[0];
  auto channels = inT->dims()[1];
  auto planes = inT->dims()[2];
  auto height = inT->dims()[3];
  auto width = inT->dims()[4];

  auto inH = inT->getHandle<float>();
  auto outH = outT->getHandle<float>();
  // elementwise product of real and imaginary planes with the mask.
  // it is just  a PoC. it can probably be just fused with fft.
  // (Evey real or imaginary output emited or not based on mask
  // also SIMD mask's can be used ot skip when fft is vectorized.

  for (size_t image = 0; image < images; image++) {
    for (size_t channel = 0; channel < channels; channel++) {
      for (size_t plane = 0; plane < planes; plane++) {
        for (size_t i = 0; i < height; i++) {
          for (size_t j = 0; j < width; j++) {
            std::array<dim_t, 5> pos = {image, channel, plane, i, j};
            auto mask = denoiseMask[i * width + j];
            outH.at(pos) = inH.at(pos) * mask;
          }
        }
      }
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
