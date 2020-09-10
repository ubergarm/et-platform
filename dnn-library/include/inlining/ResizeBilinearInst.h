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

#ifndef _RESIZE_BILINEAR_H_
#define _RESIZE_BILINEAR_H_

#include <limits>
#include <assert.h>
#include <algorithm>
#include <cmath>
#include <math.h>

#include "utils.h"
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Resize the input tensor. In general, it calculates every value in the 
 * output tensor as a weighted average of neighborhood (a.k.a. sampling 
 * locations) in the input tensor. Each dimension value of the output tensor is: 
 * output_dimension = floor(input_dimension * (roi_end - roi_start) * scale) 
 *if input "sizes" is not specified.
 * The "linear" mode includes linear interpolation for 1D tensor and N-linear 
 * interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor).
 *
 * Supported ElemKind: FloatTy, Float16Ty, Int8QTy, Int16QTy, Int32QTy, Int32ITy,
 *                     Int64ITy
 * Following InstGen.cpp Interpreter.cpp and isOpSupported at ETSOC.cpp specification.
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected.
 * @param[in] inT LibTensor input. It keeps the inputs to being handle.
 * @param[in] rszBlScale This array keeps the scale to be applied at each dimension.
 * @param[flags] flags Gives the information of the Active Shires and the
 * type of evict required.
 */

template <ElemKind elKind, size_t N>
inline typename std::enable_if_t<(isQuantizedElemKind(elKind)||(elKind==Float16Ty)), void> 
fwdLibResizeBilinearInst(LibTensor* outT, LibTensor* dataT, 
            const std::array<float, N> &rszBlScale,
            uint64_t flags, const uint32_t minionOffset = 0,
            const uint32_t assignedMinions = 0) {


  if (get_minion_id() != minionOffset) return;

  assert(dataT->getElementType() == outT->getElementType());

  /* Scaling batch and channel not supported. */
  assert(rszBlScale[0] == 1.0); 
  assert(rszBlScale[3] == 1.0);

  using elkType = typename elemKind2elemTy<elKind>::type;

  auto dataH = dataT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();
 
  float invRszBlScale_1 = 0.0;
  float invRszBlScale_2 = 0.0;
  fpReciprocalSingleElement(rszBlScale[1], invRszBlScale_1 );
  fpReciprocalSingleElement(rszBlScale[2], invRszBlScale_2);

  // Number of samples
  for (size_t ob = 0; ob < outT->dims()[0]; ++ob) {
    // Height
    for (size_t oh = 0; oh < outT->dims()[1]; ++oh) {
      // Width
      for (size_t ow = 0; ow < outT->dims()[2]; ++ow) {

  float ihf = ((oh * 1.0) * invRszBlScale_1);
  float iwf = ((ow * 1.0)  * invRszBlScale_2);
  size_t ih = static_cast<uint32_t>(ihf);
  size_t iw = static_cast<uint32_t>(iwf);

  auto ih0 = std::min(static_cast<size_t>(ih), dataT->dims()[1] - 1);
  auto ih1 = std::min(static_cast<size_t>(ih +1), dataT->dims()[1] -1);
  auto iw0 = std::min(static_cast<size_t>(iw), dataT->dims()[2] - 1);
  auto iw1 = std::min(static_cast<size_t>(iw + 1), dataT->dims()[2] -1);

  // Number of Channels
  for (size_t oc = 0; oc < outT->dims()[3]; ++oc) {

    float dst00, dst01, dst10, dst11;
    if (elKind == Float16Ty) {
      convertFp16ToFp32(static_cast<uint16_t>(dataH.at(std::array<size_t,4>{ob, ih0, iw0, oc})), dst00);
      convertFp16ToFp32(static_cast<uint16_t>(dataH.at(std::array<size_t,4>{ob, ih0, iw1, oc})), dst01);
      convertFp16ToFp32(static_cast<uint16_t>(dataH.at(std::array<size_t,4>{ob, ih1, iw0, oc})), dst10);
      convertFp16ToFp32(static_cast<uint16_t>(dataH.at(std::array<size_t,4>{ob, ih1, iw1, oc})), dst11);
    }
    else {
      dst00 = dequantize<elkType>(dataH.at(std::array<size_t,4>{ob, ih0, iw0, oc}), dataH.getScale(), dataH.getOffset());
      dst01 = dequantize<elkType>(dataH.at(std::array<size_t,4>{ob, ih0, iw1, oc}), dataH.getScale(), dataH.getOffset());
      dst10 = dequantize<elkType>(dataH.at(std::array<size_t,4>{ob, ih1, iw0, oc}), dataH.getScale(), dataH.getOffset());
      dst11 = dequantize<elkType>(dataH.at(std::array<size_t,4>{ob, ih1, iw1, oc}), dataH.getScale(), dataH.getOffset());
    }
    float hd = static_cast<float>(dst00) + static_cast<float>(dst10 - dst00) * (ihf -ih);
    float hw = static_cast<float>(dst01) + static_cast<float>(dst11 - dst01) * (ihf- ih);
    float result = hd + (hw -hd) * (iwf - iw);

    if (elKind == Float16Ty) {
      uint16_t out16 = 0;
      convertFp32ToFp16(result, out16);
      outH.at(std::array<size_t,4>{ob, oh, ow, oc}) = out16;
    }
    else {
      outH.at(std::array<size_t,4>{ob, oh, ow, oc}) = quantize<elkType>(result, outT->getScale(), outT->getOffset());
    }
  }
      }
    }
  }
}

template <ElemKind elKind, size_t N>
inline typename std::enable_if_t<(!isQuantizedElemKind(elKind) && (elKind != Float16Ty) && (elKind != BoolTy)), void>
fwdLibResizeBilinearInst(LibTensor* outT, LibTensor* dataT, 
            const std::array<float, N> &rszBlScale,
            uint64_t flags, const uint32_t minionOffset = 0,
            const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(dataT->getElementType() == outT->getElementType());

  /* Scaling batch and channel not supported. */
  assert(rszBlScale[0] == 1.0); 
  assert(rszBlScale[3] == 1.0);

  using elkType = typename elemKind2elemTy<elKind>::type;

  auto dataH = dataT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();
 
  float invRszBlScale_1 = 0.0;
  float invRszBlScale_2 = 0.0;
  fpReciprocalSingleElement(rszBlScale[1], invRszBlScale_1 );
  fpReciprocalSingleElement(rszBlScale[2], invRszBlScale_2);
  // Number of samples
  for (size_t ob = 0; ob < outT->dims()[0]; ++ob) {
    // Height
    for (size_t oh = 0; oh < outT->dims()[1]; ++oh) {
      // Width
      for (size_t ow = 0; ow < outT->dims()[2]; ++ow) {

  float ihf = ((oh * 1.0) * invRszBlScale_1);
  float iwf = ((ow * 1.0) * invRszBlScale_2);

  /* @TODO Due to SW-1974 Change to uint64_t once ticket will be solved.*/
  size_t ih = static_cast<uint32_t>(ihf);
  size_t iw = static_cast<uint32_t>(iwf);

  auto ih0 = std::min(static_cast<size_t>(ih), dataT->dims()[1] - 1);
  auto ih1 = std::min(static_cast<size_t>(ih +1), dataT->dims()[1] -1);
  auto iw0 = std::min(static_cast<size_t>(iw), dataT->dims()[2] - 1);
  auto iw1 = std::min(static_cast<size_t>(iw + 1), dataT->dims()[2] -1);

  // Number of Channels
  for (size_t oc = 0; oc < outT->dims()[3]; ++oc) {
    auto v00 = dataH.at(std::array<size_t,4>{ob, ih0, iw0, oc});
    auto v01 = dataH.at(std::array<size_t,4>{ob, ih0, iw1, oc});
    auto v10 = dataH.at(std::array<size_t,4>{ob, ih1, iw0, oc});
    auto v11 = dataH.at(std::array<size_t,4>{ob, ih1, iw1, oc});

    auto hd = static_cast<float>(v00) + static_cast<float>(v10 - v00) * (ihf -ih);
    auto hw = static_cast<float>(v01) + static_cast<float>(v11 -v01) * (ihf- ih);
    float result = hd + (hw -hd) * (iwf - iw);
    outH.at(std::array<size_t,4>{ob, oh, ow, oc}) = result;
  }
      }
    }
  }
}

} // inlining
} // dnn_lib

#endif
