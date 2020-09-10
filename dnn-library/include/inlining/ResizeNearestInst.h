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

#include "utils.h"
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

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
inline typename std::enable_if_t<(elKind != BoolTy), void>
fwdLibResizeNearestInst(LibTensor* outT, LibTensor* inT, 
      const std::array<float, N> &rszScale, 
      uint64_t flags, const uint32_t minionOffset = 0, 
      const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(inT->getElementType() == outT->getElementType());

  using elkType = typename elemKind2elemTy<elKind>::type; 

  auto inH = inT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();

  for (size_t ob = 0; ob < outT->dims()[0]; ++ob) {
    auto ib = std::min(size_t(ob / rszScale[0]), inT->dims()[0] - 1);
    for (size_t oh = 0; oh < outT->dims()[1]; ++oh) {
      auto ih = std::min(size_t(oh / rszScale[1]), inT->dims()[1] - 1);
      for (size_t ow = 0; ow < outT->dims()[2]; ++ow) {
        auto iw = std::min(size_t(ow / rszScale[2]), inT->dims()[2] - 1);
        for (size_t oc = 0; oc < outT->dims()[3]; ++oc) {
          auto ic = std::min(size_t(oc / rszScale[3]), inT->dims()[3] - 1);
          outH.at(std::array<size_t,4>{ob, oh, ow, oc}) = 
      inH.at(std::array<size_t,4>{ib, ih, iw, ic});
        }
      }
    }
  }

  outT->evict(DO_EVICTS);
}

}  // inlining
}  // dnn_space

#endif
