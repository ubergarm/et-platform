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

#ifndef _IMAGE_CONDITIONING_H_
#define _IMAGE_CONDITIONING_H_

#include "LibTensor.h"
#include "utils.h"
#include <algorithm>
#include <assert.h>
#include <limits>

namespace dnn_lib {

namespace inlining {

template <class T> static INLINE_ATTR T conditionPixel(float inputPixel, size_t channel) {

  constexpr auto min = float(std::numeric_limits<uint8_t>::min());
  constexpr auto max = float(std::numeric_limits<uint8_t>::max());

  // These are standard normalization factors for imagenet, adjusted for
  // normalizing values in the 0to255 range instead of 0to1, as seen at:
  // https://github.com/pytorch/examples/blob/master/imagenet/main.py

  constexpr float mean[] = {0.485f * max, 0.456f * max, 0.406f * max};
  // using reciprocal to avoid fp div, scaling 0..1 on the same shot.
  constexpr float stdevAndScaleRec[] = {1 / (0.229f * max), 1 / (0.224f * max), 1 / (0.225f * max)};
  auto val = std::clamp(inputPixel, min, max);
  val -= mean[channel];
  val *= stdevAndScaleRec[channel];

  return val;
}

__attribute__((noinline)) static void resnetImageCondition(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                                           const uint32_t minionOffset, size_t numMinions,
                                                           size_t minionId);
__attribute__((noinline)) static void resnetImageCondition(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                                           const uint32_t minionOffset, size_t numMinions,
                                                           size_t minionId) {
  //  FIXME: just minon 0 does some work at the moment.
  if ((minionId - minionOffset) != 0) {
    return;
  }

  et_printf("%s(%d) [%d]\n", __func__, __LINE__, minionId);

  (void)flags;
  (void)minionOffset;
  (void)numMinions;

  // Receives an input tensor of
  //  float <batch x channels x 2 x 256 x 256 >
  // And collapses to
  //  float <batch x channels x 224 x 224>
  // Also values are clipped 0 .. 255 as IFFT may produce invalid values due to precision.
  // Other steps
  // imagenet dataset normalizaton
  // 0.. to 1 encoding
  // int8 quantize (depending on config)

  auto inImages = inT->dims()[0];
  auto inChannels = inT->dims()[1];
  auto inPlanes = inT->dims()[2];
  auto inHeight = inT->dims()[3];
  auto inWidth = inT->dims()[4];

  auto outImages = outT->dims()[0];
  auto outChannels = outT->dims()[1];
  auto outHeight = outT->dims()[2];
  auto outWidth = outT->dims()[3];

  et_assert(inImages == outImages);
  et_assert(inChannels == outChannels);
  et_assert(inHeight >= outHeight);
  et_assert(inWidth >= outWidth);
  et_assert(inPlanes == 2);
  auto inH = inT->getHandle<float>();
  auto outH = outT->getHandle<float>();

  for (size_t image = 0; image < outImages; image++) {
    for (size_t i = 0; i < outHeight; i++) {
      for (size_t j = 0; j < outWidth; j++) {
        // rgb to bgr TODO: find faster ways if needed: (transpose, slicing,...)
        std::array<dim_t, 5> inPosR = {image, 0, 0, i, j};
        std::array<dim_t, 5> inPosG = {image, 1, 0, i, j};
        std::array<dim_t, 5> inPosB = {image, 2, 0, i, j};
        std::array<dim_t, 4> outPosR = {image, 0, i, j};
        std::array<dim_t, 4> outPosG = {image, 1, i, j};
        std::array<dim_t, 4> outPosB = {image, 2, i, j};

        outH.at(outPosB) = conditionPixel<float>(inH.at(inPosR), 0);
        outH.at(outPosG) = conditionPixel<float>(inH.at(inPosG), 1);
        outH.at(outPosR) = conditionPixel<float>(inH.at(inPosB), 2);
      }
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _IMAGE_CONDITIONING_H_
