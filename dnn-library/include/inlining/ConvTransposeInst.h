/*-------------------------------------------------------------------------
 * Copyright (C) 2019, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _CONVOLUTION_TRANSPOSE_INST_H_
#define _CONVOLUTION_TRANSPOSE_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>
#include <math.h>

#include "Float16.h"
#include "utils.h" // From include/internal path
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

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
template <ElemKind dstElK, size_t N, size_t PN>
inline void fwdLibConvTransposeInst(LibTensor* outT, LibTensor* dataT, 
            LibTensor* filterT, LibTensor* biasT,
            const std::array<uint32_t, N> &kernels,
            const std::array<uint32_t, N> &strides,
            const std::array<uint32_t, PN> &pads,
            const uint32_t group, 
            const uint32_t dilation,
            const uint64_t flags, 
            const uint32_t minionOffset = 0, 
            const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(dstElK == FloatTy);
  assert(outT->getElementType() == dataT->getElementType());
  assert(dataT->getElementType() == filterT->getElementType());

  /* ob #samples, oh height, ow witdh, oc #channels*/
  /* data and output channel must be divisible by group. */
  assert((dataT->dims()[3] % group)==0);
  assert((outT->dims()[3] % group)==0);
  assert(group == 1); //group must be 1

  size_t inCperG = (dataT->dims()[3] / group);
  size_t outCperG = (outT->dims()[3] / group);

  using elkType = typename elemKind2elemTy<dstElK>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = dataT->getHandle<elkType>();
  auto filterH = filterT->getHandle<elkType>();
  auto biasH = biasT->getHandle<elkType>();

  // For each input in the bach
  for (size_t n = 0; n < dataT->dims()[0]; n++) {
    
    //Init bias, @TODO take out to a separate function when quant is in.
    for (size_t ax = 0; ax < outT->dims()[1]; ax++) {
      for (size_t ay = 0; ay < outT->dims()[2]; ay++) {
  for (size_t d = 0; d < outT->dims()[3]; d++) {
    outH.at(std::array<size_t, 4>{n, ax, ay, d}) = static_cast<elkType>(biasH.at(std::array<size_t, 1>{d}));
  }
      }
    }

    //For each group of input channels
    for (size_t g = 0; g < group; g++) {
      //For each input channel in the group:
      for (size_t d = (g * inCperG); d < ((g + 1) * inCperG); d++) {
  //For each transposed convolution 'jump' in the input tnesor:
  ssize_t x = -static_cast<ssize_t>(pads[0]); //near
  for (size_t bx = 0; bx < dataT->dims()[1]; bx++, x += strides[0]) {
    ssize_t y = -static_cast<ssize_t>(pads[1]);
    for (size_t by = 0; by < dataT->dims()[2]; by++, y += strides[1]) {
      //For each element in the each transposed convolution filter:
      elkType input = dataH.at(std::array<size_t, 4>{n, bx, by, d});

      for (size_t kx = 0; kx < kernels[0]; kx++) {
        for (size_t ky = 0; ky < kernels[1]; ky++) {
    ssize_t ax = x + kx * dilation;
    ssize_t ay = y + ky * dilation;

    //Ignore index access below zero (this is due to padding).
    if (ax < 0 || ay < 0 || ax >= ssize_t(outT->dims()[1]) ||
        ay >= ssize_t(outT->dims()[2])) {
      continue;
    }
    
    for (size_t c = 0; c < outCperG; c++) {
      outH.at(std::array<size_t, 4>{n, static_cast<size_t>(ax), static_cast<size_t>(ay), (g * outCperG + c)}) +=
        filterH.at(std::array<size_t,4>{c, kx, ky, d}) * input;
    }
        }
      }
    }
  }
      }
    }
  }


}

} // inlining
} // name_space

#endif
