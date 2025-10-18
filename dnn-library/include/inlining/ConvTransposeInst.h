/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _CONVOLUTION_TRANSPOSE_INST_H_
#define _CONVOLUTION_TRANSPOSE_INST_H_

#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <math.h>
#include <string.h>

#include "Float16.h"
#include "LibCommon.h"
#include "LibTensor.h"
#include "LibTypes.h"
#include "utils.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief 
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected data.
 * @param[in] dataT LibTensor input. It keeps the Data.
 * @param[in] filterT LibTensor input. It keeps filter data.
 * @param[in] biasT LibTensor input. It keeps bias data.
 * @param[in] kernels
 * @param[in] strides
 * @param[in] pads
 * @param[in] group
 * @param[in] dilation
 * @param[flags] flags Gives the information of the Active Shires and the
 *  type of evict required.
 */
template <ElemKind dstElK, size_t N, size_t PN, size_t KN>
inline void fwdLibConvTransposeInst(LibTensor* outT, LibTensor* dataT, LibTensor* filterT, LibTensor* biasT,
                                    const std::array<uint32_t, N>& kernels, const std::array<uint32_t, N>& strides,
                                    const std::array<uint32_t, PN>& pads, const uint32_t group,
                                    const std::array<uint32_t, KN>& dilation, [[maybe_unused]] const uint64_t flags,
                                    const uint32_t minionOffset = 0,
                                    [[maybe_unused]] const uint32_t assignedMinions = 0) {
  using elkType = typename elemKind2elemTy<dstElK>::type;

  et_assert(get_minion_id() >= minionOffset);

  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  assert(dstElK == FloatTy);
  assert(outT->getElementType() == dataT->getElementType());
  assert(dataT->getElementType() == filterT->getElementType());

  /* ob #samples, oh height, ow witdh, oc #channels*/
  /* data and output channel must be divisible by group. */
  assert((dataT->dims()[3] % group)==0);
  assert((outT->dims()[3] % group)==0);
  assert(group == 1); //group must be 1

  const dim_t* actIndex = outT->dims().data();
  const dim_t* dstPitch = outT->strides().data();
  void* dst = outT->getRawDataPointer();
  auto numElemsDst = dstPitch[0] * actIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<elkType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dst);
  if (maxRead == 0)
    return;

  size_t inCperG = (dataT->dims()[3] / group);
  size_t outCperG = (outT->dims()[3] / group);

  auto outH = outT->getHandle<elkType>();
  auto dataH = dataT->getHandle<elkType>();
  auto filterH = filterT->getHandle<elkType>();
  auto biasH = biasT->getHandle<elkType>();

  auto inpDims = dataT->dims();

  // We move the initialAddr to the next non-padding position
  dim_array_t coord = {0}; // Vector of coordinates
  dim_t k = 0;             // Amount of non-zero coordinates
  dim_t srcDimNum = dataT->ndims();
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex, k);

  uint64_t offsetOut = 0;
  for (dim_t j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }
  size_t posMax = maxRead + initialAddr;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    auto outCoord = std::array<size_t, 4>{coord[0], coord[1], coord[2], coord[3]};
    outH.at(outCoord) = biasH.at(std::array<size_t, 1>{coord[3]});
    size_t n = coord[0];
    size_t ax = coord[1];
    size_t ay = coord[2];
    size_t c = coord[3] % outCperG;
    size_t g = coord[3] / outCperG;
    for (size_t d = (g * inCperG); d < ((g + 1) * inCperG); d++) {
      for (size_t kx = 0; kx < kernels[0]; kx++) {
        for (size_t ky = 0; ky < kernels[1]; ky++) {
          ssize_t x = ax - kx * dilation[0];
          ssize_t y = ay - ky * dilation[1];
          if ((((x + pads[0]) % strides[0]) != 0) or (((y + pads[1]) % strides[1]) != 0)) {
            continue;
          }
          size_t bx = (x + pads[0]) / strides[0];
          size_t by = (y + pads[1]) / strides[1];
          if (((int(x) + int(pads[0])) < 0) or (bx >= inpDims[1]) or ((int(y) + int(pads[1])) < 0) or
              (by >= inpDims[2])) {
            continue;
          }
          auto inpCoord = std::array<size_t, 4>{n, bx, by, d};
          auto filCoord = std::array<size_t, 4>{c, kx, ky, d};
          outH.at(outCoord) += dataH.at(inpCoord) * filterH.at(filCoord);
        }
      }
    }
    done = getOffsets(srcDimNum, coord, offsetOut, actIndex, dstPitch);
  }
}

} // inlining
} // name_space

#endif
