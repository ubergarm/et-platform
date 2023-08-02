/*-------------------------------------------------------------------------
 * Copyright (C) 2020, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _RESIZE_NEAREST_H_
#define _RESIZE_NEAREST_H_

#include <limits>
#include <assert.h>

#include "LibCommon.h"
#include "LibTensor.h"
#include "LibTypes.h"
#include "utils.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Resizes Generate an Output tensor with the spatial dimensions of the input
 * using nearest neighbor interpolation. The width_scale and height_scale arguments
 * control the size of the output, which is given by:
 * output_width = floor(input_width * width_scale)
 * output_height = floor(output_height * height_scale)
 *
 * BoolTy and Fused kinds not supported
 * Following InstGen.cpp Interpreter.cpp and isOpSupported at ETSOC.cpp specification.
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected.
 * @param[in] inT LibTensor input. It keeps the inputs to being handle.
 * @param[in] rszScale This array keeps the scale to be applied at each dimension.
 * @param[flags] flags Gives the information of the Active Shires and the
 * type of evict required.
 */

template <ElemKind elKind, size_t N>
INLINE_ATTR typename std::enable_if_t<(elKind != BoolTy), void>
fwdLibResizeNearestInst(LibTensor* outT, LibTensor* inT, const std::array<float, N>& rszScale, uint64_t flags,
                        const uint32_t minionOffset = 0, [[maybe_unused]] const uint32_t assignedMinions = 0) {

  using elkType = typename elemKind2elemTy<elKind>::type;

  et_assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  auto inH = inT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();

  et_assert(inT->getElementType() == outT->getElementType());

  std::array<float, N> invRszScale = {
    0.0,
  };

  for (size_t i = 0; i < N; i++) {
    getReciprocal(rszScale[i], invRszScale[i]);
  }

  void* dst = outT->getRawDataPointer();

  const dim_t* actIndex = outT->dims().data();
  const dim_t* dstPitch = outT->strides().data();

  dim_t srcDimNum = inT->ndims();

  auto numElemsDst = dstPitch[0] * actIndex[0];

  size_t initialAddr, maxRead;
  size_t typeSize = getsize<elkType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dst);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position
  dim_array_t coord = {0}; // Vector of coordinates
  dim_t k = 0;             // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex, k);

  // We get the actual initialAddr, in the input and output.
  uint64_t offsetOut = 0;
  for (dim_t j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }
  size_t posMax = maxRead + initialAddr;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    // We get first coordinate (batch)
    auto x = static_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(coord[0])) * invRszScale[0]);
    auto xx = static_cast<uint32_t>(inT->dims()[0] - 1);
    auto ib = std::min(x, xx);
    // We get second coordinate (height)
    auto y = static_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(coord[1])) * invRszScale[1]);
    auto yy = static_cast<uint32_t>(inT->dims()[1] - 1);
    auto ih = std::min(y, yy);
    // We get third coordinate (width)
    auto t = static_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(coord[2])) * invRszScale[2]);
    auto tt = static_cast<uint32_t>(inT->dims()[2] - 1);
    auto iw = std::min(t, tt);
    // We get forth coordinate (channels)
    auto z = static_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(coord[3])) * invRszScale[3]);
    auto zz = static_cast<uint32_t>(inT->dims()[3] - 1);
    auto ic = std::min(z, zz);

    outH.at(std::array<size_t, 4>{coord[0], coord[1], coord[2], coord[3]}) =
      inH.at(std::array<size_t, 4>{ib, ih, iw, ic});
    done = getOffsets(srcDimNum, coord, offsetOut, actIndex, dstPitch);
  }

  outT->evict(DO_EVICTS);
}

}  // inlining
}  // dnn_space

#endif
