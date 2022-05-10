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

template <bool inverse = false>
INLINE_ATTR void fft(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset,
                     const uint32_t numMinions, uint32_t minionId) {

  (void)flags;  
  (void)numMinions;

  et_printf("%s(%d) [%d]\n", __func__, __LINE__, minionId);

  float* in = inT->getRawDataPointer<float>();
  float* out = outT->getRawDataPointer<float>();

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
  constexpr size_t workRowBits = 0;
  constexpr size_t workRowBranchBits = 0;
  constexpr size_t workColBits = 0;
  constexpr size_t workColBranchBits = 0;
  static_assert(workRowBits + workRowBranchBits ==  workColBits + workColBranchBits);      

  assert(numMinions == 1 << (workRowBits + workRowBranchBits));
  if ((minionId - minionOffset) >= numMinions) {
    return;
  }

  for (size_t batch = 0; batch < batches; ++batch) {
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
        fft2d_inv_threaded<workRowBits, workRowBranchBits, workColBits, workColBranchBits, pass1, pass2> (
          minionOffset, minionId, width, height, real, real_stride, img, img_stride, result_real, result_real_stride, result_img,
          result_img_stride);
      } else {
        fft2d_threaded<workRowBits, workRowBranchBits, workColBits, workColBranchBits>(
          minionOffset, minionId, width, height, real, real_stride, img, img_stride, result_real, result_real_stride, result_img,
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

  // If Minion is outside the group assigned to this Node get out.
  auto minionId = get_minion_id();
  if ((minionId - minionOffset) >= numMinions) {
    return;
  }

  et_printf("%s(%d) [%d] called with op: %s \n", __func__, __LINE__, minionId, Op2String[op]);

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
