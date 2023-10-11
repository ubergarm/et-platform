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
#include "etsoc/common/utils.h"
#include "utils.h"
#include <algorithm>
#include <assert.h>
#include <limits>

namespace dnn_lib {

namespace inlining {

template <class T, size_t channel> static INLINE_ATTR T conditionPixel(float inputPixel) {

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

  (void)flags;

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

  size_t relativeMinionId = minionId - minionOffset;

  // Assert the ad-hoc tiling applies
  et_assert(outWidth == 224);
  et_assert(outHeight == 224);
  et_assert(numMinions == 1024 or numMinions == 256);

  size_t colBits;
  if (numMinions == 256) {
    colBits = 0;
  } else {
    colBits = 2;
  }

  // Assign 4 minions per row
  size_t row0 = (relativeMinionId >> colBits);
  size_t numRows = 1;
  if (row0 >= outHeight) {
    return;
  }
  size_t lastRow = row0 + numRows;

  // Split the row between the assigned minions (either 1 or 4 minions)
  size_t line0;
  size_t lines;
  if (numMinions == 256) {
    // 14 lines per minion
    line0 = 0;
    lines = 14;
  } else {
    // Split a row between the 4 assigned minions
    switch (relativeMinionId & 3) {
    case 0:
      line0 = 0;
      lines = 4;
      break;
    case 1:
      line0 = 4;
      lines = 4;
      break;
    case 2:
      line0 = 8;
      lines = 3;
      break;
    case 3:
      line0 = 11;
      lines = 3;
      break;
    default:
      et_assert(false);
      line0 = 0;
      lines = 0;
      break;
    }
  }
  constexpr size_t elemsPerCacheLine = 16;
  size_t col0 = line0 * elemsPerCacheLine;
  size_t numCols = lines * elemsPerCacheLine;
  size_t lastCol = col0 + numCols;

  auto inH = inT->getHandle<float>();
  auto outH = outT->getHandle<float>();

  for (size_t image = 0; image < outImages; image++) {
    for (size_t i = row0; i < lastRow; i++) {
      for (size_t j = col0; j < lastCol; j++) {
        // rgb to bgr TODO: find faster ways if needed: (transpose, slicing,...)
        std::array<dim_t, 5> inPosR = {image, 0, 0, i, j};
        std::array<dim_t, 5> inPosG = {image, 1, 0, i, j};
        std::array<dim_t, 5> inPosB = {image, 2, 0, i, j};
        std::array<dim_t, 4> outPosR = {image, 0, i, j};
        std::array<dim_t, 4> outPosG = {image, 1, i, j};
        std::array<dim_t, 4> outPosB = {image, 2, i, j};

        outH.at(outPosB) = conditionPixel<float, 0>(inH.at(inPosR));
        outH.at(outPosG) = conditionPixel<float, 1>(inH.at(inPosG));
        outH.at(outPosR) = conditionPixel<float, 2>(inH.at(inPosB));
      }

      // Evict to L3 the produced lines
      constexpr auto destination = cop_dest::to_L3;
      std::array<dim_t, 4> outPosR = {image, 0, i, col0};
      std::array<dim_t, 4> outPosG = {image, 1, i, col0};
      std::array<dim_t, 4> outPosB = {image, 2, i, col0};

      uintptr_t blueAddress = reinterpret_cast<uintptr_t>(&outH.at(outPosB));
      uintptr_t greenAddress = reinterpret_cast<uintptr_t>(&outH.at(outPosG));
      uintptr_t redAddress = reinterpret_cast<uintptr_t>(&outH.at(outPosR));

      evict_va_multi(destination, blueAddress, lines);
      evict_va_multi(destination, greenAddress, lines);
      evict_va_multi(destination, redAddress, lines);
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _IMAGE_CONDITIONING_H_
