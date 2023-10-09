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
#include "ImageConditioning.h"
#include "LibTensor.h"
#include "etsoc/common/utils.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

#ifdef GPSDK
#include "sync.h"
#endif

namespace dnn_lib {

namespace inlining {

enum class Operation {
  FFT = 0,
  IFFT,
  NOISE_FILTER_1,
  FFT_FILTER_FUSED,
  RESNET_CONDITION,
  RESNET_CONDITION_QUANTIZE_FUSED,
  COUNT,
};

static constexpr const char* Op2String[] = {
  "FFT", "IFFT", "NOISE_FILTER_1", "FFT_FILTER_FUSED", "CONDITION", "CONDITION_QUANTIZE_FUSED"};

template <typename T> INLINE_ATTR T min(T a, T b) {
  return (a < b) ? a : b;
}

INLINE_ATTR void fftTiling(size_t batches, [[maybe_unused]] size_t channels, [[maybe_unused]] size_t components,
                           size_t height, size_t width, size_t numMinions, size_t& workBatchBits,
                           size_t& workChannelBits, size_t& workRowBits, size_t& workRowBranchBits, size_t& workColBits,
                           size_t& workColBranchBits) {

  // About to assign all the bits
  size_t availableBits = log2(numMinions);

  // Assign as many bits as possible in the batch element dimension
  workBatchBits = min(log2(batches), availableBits);
  availableBits -= workBatchBits;

  // If 3 channels, roundup to 4
  if (channels == 3) {
    ++channels;
  }

  // Assign as many (of the remaining) bits as possible in the channel dimension
  workChannelBits = min(log2(channels), availableBits);
  availableBits -= workChannelBits;

  if (height >= width) {
    // Assign as many (of the remaining) bits as possible in the rows dimension
    workRowBits = min(log2(height), availableBits);
    availableBits -= workRowBits;

    // Assign the remaiming bits on the row branches dimension
    workRowBranchBits = min(log2(width), availableBits);

    // Assign as many (of the remaining) minion bits for the columns dimension as
    // done for the rows dimension. If the columns dimension is
    // not as big then do tiling also on branches.
    workColBits = min(log2(width), workRowBits);
    workColBranchBits = workRowBits + workRowBranchBits - workColBits;
  } else {
    // Assign as many (of the remaining) bits as possible in the rows dimension
    workColBits = min(log2(width), availableBits);
    availableBits -= workColBits;

    // Assign the remaiming bits on the row branches dimension
    workColBranchBits = min(log2(height), availableBits);

    // Assign as many (of the remaining) minion bits for the columns dimension as
    // done for the rows dimension. If the columns dimension is
    // not as big then do tiling also on branches.
    workRowBits = min(log2(height), workColBits);
    workRowBranchBits = workColBits + workColBranchBits - workRowBits;
  }

  // Overriding tiling?
  if constexpr (false) {
    workBatchBits = 0;
    workChannelBits = 0;
    workRowBits = 0;
    workRowBranchBits = 0;
    workColBits = 0;
    workColBranchBits = 0;
  }

  // Irrespective of whatever we do for tiling, the optimal configuration
  // must be produced for the denoise demo or we make it crash here rather
  // than allowing the demo to run with a regressed peformance.
  if (batches == 1 and channels == 3 and height == 256 and width == 256) {
    assert(workBatchBits == 0);
    assert(workChannelBits == 2);
    assert(workRowBits == 8);
    assert(workRowBranchBits == 0);
    assert(workColBits == 8);
    assert(workColBranchBits == 0);
  }

  assert(workRowBits + workRowBranchBits == workColBits + workColBranchBits);
  assert(workBatchBits + workRowBits + workRowBranchBits <= log2(numMinions));
}

INLINE_ATTR size_t getFilterIndex(LibTensor* inT) {
  // By convention filter_index is passed s the first 32 bits of img plane on inputTensor, (batch 0, img 0). This is
  // 0,0,1,0.0.
  auto inH = inT->getHandle<uint32_t>();
  std::array<dim_t, 5> filterIndexPos = {0, 0, 1, 0, 0};
  return size_t(inH.at(filterIndexPos));
}

template <bool inverse = false, bool freqDomainFilterFusion = false>
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
  size_t workChannelBits;
  size_t workRowBits, workRowBranchBits;
  size_t workColBits, workColBranchBits;
  fftTiling(batches, channels, components, height, width, numMinions, workBatchBits, workChannelBits, workRowBits,
            workRowBranchBits, workColBits, workColBranchBits);

  if (minionId - minionOffset == 0) {
#ifdef ENABLE_TRACES
    et_printf("bBits=%d chBits=%d rBits=%d rBrBits=%d cBits=%d cBrBits=%d", workBatchBits, workChannelBits, workRowBits,
              workRowBranchBits, workColBits, workColBranchBits);
#endif
  }

  // Ensure we got assigned at least as many minions as we can use
  assert(numMinions >= static_cast<size_t>(1 << (workBatchBits + workChannelBits + workRowBits + workRowBranchBits)));

  // Use just as many minions as we can
  size_t usedMinions = min(numMinions, 1UL << (workBatchBits + workChannelBits + workRowBits + workRowBranchBits));

  // Only used minions get work
  if (minionId - minionOffset < usedMinions) {

    // et_printf("%s(%d) [minionOffset=%d numMinions=%d minionId=%d]\n", __func__, __LINE__, minionOffset, numMinions,
    // minionId);

    float* in = inT->getRawDataPointer<float>();
    float* out = outT->getRawDataPointer<float>();

    size_t batchElemsGroupSize = 1 << workBatchBits;
    size_t channelElemsGroupSize = 1 << workChannelBits;

    // get filterIndex from host.
    [[maybe_unused]] auto filterIndex = getFilterIndex(inT);

    for (size_t batch0 = 0; batch0 < batches; batch0 += batchElemsGroupSize) {

      // Get the batch group id based on the top 'workBatchBits', ignoring the rest
      size_t batchMinionGroupId =
        (minionId & ((1 << (workBatchBits + workChannelBits + workColBits + workColBranchBits)) - 1)) >>
        (workChannelBits + workColBits + workColBranchBits);

      // Get which batch this minion will perform computation on
      size_t batch = batch0 + batchMinionGroupId;

      size_t minionOffset0 = (minionOffset + minionId) & ~((1 << (workColBits + workColBranchBits)) - 1);
      size_t minionId0 = minionId & ((1 << (workColBits + workColBranchBits)) - 1);

      // et_printf("%s(%d) [mId=%d nMins=%d mOfs0=%d mId0=%d batch=%d]\n", __func__, __LINE__, minionId, numMinions,
      //          minionOffset0, minionId0, batch);

      for (size_t channel0 = 0; channel0 < channels; channel0 += channelElemsGroupSize) {

        // Get the channel group id based only on the 'workChannelBits', ignoring the rest
        size_t channelMinionGroupId = (minionId & ((1 << (workChannelBits + workColBits + workColBranchBits)) - 1)) >>
                                      (workColBits + workColBranchBits);

        // Get which channel the minion will perform computation on
        size_t channel = channel0 + channelMinionGroupId;

        if (channel >= channels) {
          // Checking this condition is needed because channels get rounded up
          continue;
        }

        // Get pointers to input and output data structures based on 'batch' and 'channel'
        float* real = in + srcStrides[0] * batch + srcStrides[1] * channel;
        size_t realStride = srcStrides[3];
        float* img = real + srcStrides[2];
        size_t imgStride = srcStrides[3];
        float* resultReal = out + dstStrides[0] * batch + dstStrides[1] * channel;
        size_t resultRealStride = dstStrides[3];
        float* resultImg = resultReal + dstStrides[2];
        size_t resultImgStride = dstStrides[3];

        constexpr bool pass1 = true;
        constexpr bool pass2 = true;

        if constexpr (inverse) {
          fft2DInvThreaded<pass1, pass2>(workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset,
                                         minionOffset0, minionId0, width, height, real, realStride, img, imgStride,
                                         resultReal, resultRealStride, resultImg, resultImgStride);
        } else {
          fft2dThreaded<pass1, pass2, false, freqDomainFilterFusion>(
            workRowBits, workRowBranchBits, workColBits, workColBranchBits, minionOffset, minionOffset0, minionId0,
            width, height, real, realStride, img, imgStride, resultReal, resultRealStride, resultImg, resultImgStride,
            filterIndex);
        }
      }
    }
  }

  // Synchronize all minions, for when using some shire only partially
  constexpr size_t flb = 31;
  constexpr size_t thread = 0;
  constexpr size_t fcc = 0;
  constexpr size_t allMinionsMask = (1ULL << SOC_MINIONS_PER_SHIRE) - 1;
  // Minion zero sends a credit to all the minions in the shire
  if (flbarrier(flb, SOC_MINIONS_PER_SHIRE - 1)) {
    fcc_send(SHIRE_OWN, thread, fcc, allMinionsMask);
  }
  fcc_consume(fcc);

#ifdef GPSDK
  hart::barrier();
#else
  // Synchronize all the shires, for when not using them all
  barrier(minionOffset, numMinions, 1);
#endif
}

INLINE_ATTR void freqDomainNoiseFilter(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset,
                                       size_t numMinions, size_t minionId) {
  //  FIXME: just minon 0 does some work at the moment.
  if ((minionId - minionOffset) != 0) {
    return;
  }

  // et_printf("%s(%d) [%d]\n", __func__, __LINE__, minionId);

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

  auto filterIndex = getFilterIndex(inT);

  for (size_t image = 0; image < images; image++) {
    for (size_t channel = 0; channel < channels; channel++) {
      for (size_t plane = 0; plane < planes; plane++) {
        for (size_t i = 0; i < height; i++) {
          for (size_t j = 0; j < width; j++) {
            std::array<dim_t, 5> pos = {image, channel, plane, i, j};
            auto mask = denoiseMask[filterIndex][i * width + j];
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
#ifdef ENABLE_TRACES
    et_printf("%s(%d) [numMinions=%d op=%d]\n", __func__, __LINE__, numMinions, op);
#endif
  }

  switch (Operation(op)) {
  case Operation::FFT:
    return fft<false>(outT, inT, flags, minionOffset, numMinions, minionId);
  case Operation::FFT_FILTER_FUSED:
    return fft<false, true>(outT, inT, flags, minionOffset, numMinions, minionId);
  case Operation::IFFT:
    return fft<true>(outT, inT, flags, minionOffset, numMinions, minionId);
  case Operation::NOISE_FILTER_1:
    return freqDomainNoiseFilter(outT, inT, flags, minionOffset, numMinions, minionId);
  case Operation::RESNET_CONDITION:
    return resnetImageCondition(outT, inT, flags, minionOffset, numMinions, minionId);
  case Operation::RESNET_CONDITION_QUANTIZE_FUSED:
  case Operation::COUNT:
  default:
    et_assert("unsupported operation");
  }

  return;
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ETSOCGENERICOP_INST_H_
