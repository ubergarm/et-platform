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
#include "LibTensor.h"
#include "utils.h"
#include "FFT.h"
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
  LAST = COUNT - 1
};

static constexpr const char* Op2String[] = {"FFT", "IFFT", "NOISE_FILTER_1"};

template<bool inverse = false>
INLINE_ATTR void fft(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset,
                     const uint32_t assignedMinions, uint32_t activeMinions, uint32_t minionId) {

  //  FIXME: just minon 0 does some work at the moment.
  if (minionId != 0) {
    return;
  }
  (void)flags;
  (void)minionOffset;
  (void)activeMinions;
  (void)assignedMinions;

  et_printf("%s(%d) [%d]\n", __func__, __LINE__, minionId);

  float* in = inT->getRawDataPointer<float>();
  float* out = outT->getRawDataPointer<float>();

  const dim_t *srcDims = inT->dims().data();
  const dim_t *srcStrides = inT->strides().data();
  const dim_t *dstStrides = outT->strides().data();

  size_t batches = srcDims[0];

  for (size_t batch = 0; batch < batches; ++batch) {
    size_t width = srcDims[3];
    size_t height = srcDims[2];
    float* real = in + srcStrides[0] * batch;
    size_t real_stride = srcStrides[2];
    float* img = real + srcStrides[1];
    size_t img_stride = srcStrides[2];
    float* result_real = out + dstStrides[0] * batch;
    size_t result_real_stride = dstStrides[2];
    float* result_img = result_real + dstStrides[1];
    size_t result_img_stride =  dstStrides[2];

    if constexpr (inverse) {
      fft2d_inv(width, height, real, real_stride,
                   img, img_stride, result_real,
                   result_real_stride, result_img,
                   result_img_stride);
    } else {
      fft2d(width, height, real, real_stride,
                   img, img_stride, result_real,
                   result_real_stride, result_img,
                   result_img_stride);
    }

  }
}

INLINE_ATTR void freqDomainNoiseFilter(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset,
                                       const uint32_t assignedMinions, uint32_t activeMinions, uint32_t minionId) {
  //  FIXME: just minon 0 does some work at the moment.
  if (minionId != 0) {
    return;
  }

  et_printf("%s(%d) [%d]\n", __func__, __LINE__, minionId);

  (void)flags;
  (void)minionOffset;
  (void)activeMinions;
  (void)assignedMinions;

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

  (void)flags;

  static_assert(elK == FloatTy);

  et_printf("%s(%d) [%d] called with op: %s \n", __func__, __LINE__, get_minion_id(), Op2String[op]);

  auto minionId = get_minion_id();
  // Rebase minion ID.
  minionId -= minionOffset;

  // Get number of Minions assigned to this Node.
  uint64_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;

  // If Minion is outside the group assigned to this Node get out.
  if (minionId >= activeMinions) {
    return;
  }

  switch (Operation(op)) {
  case Operation::FFT:
    return fft<false>(outT, inT, flags, minionOffset, assignedMinions, activeMinions, minionId);
  case Operation::IFFT:
    return fft<true>(outT, inT, flags, minionOffset, assignedMinions, activeMinions, minionId);
  case Operation::NOISE_FILTER_1:
    return freqDomainNoiseFilter(outT, inT, flags, minionOffset, assignedMinions, activeMinions, minionId);
  default:
    et_assert("unsupported operation");
  }

  return;
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ETSOCGENERICOP_INST_H_
