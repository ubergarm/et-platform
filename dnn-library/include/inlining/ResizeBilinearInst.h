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

#include "LibCommon.h"
#include "LibTensor.h"
#include "LibTypes.h"
#include "utils.h"
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <limits>
#include <math.h>

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
INLINE_ATTR typename std::enable_if_t<(isQuantizedElemKind(elKind) || (elKind == Float16Ty)), void>
fwdLibResizeBilinearInst(LibTensor* outT, LibTensor* dataT, const std::array<float, N>& rszBlScale,
                         [[maybe_unused]] uint64_t flags, const uint32_t minionOffset = 0,
                         [[maybe_unused]] const uint32_t assignedMinions = 0) {
  using elkType = typename elemKind2elemTy<elKind>::type;

  et_assert(get_minion_id() >= minionOffset);

  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  assert(dataT->getElementType() == outT->getElementType());

  /* Scaling batch and channel not supported. */
  assert(rszBlScale[0] == 1.0f);
  assert(rszBlScale[3] == 1.0f);

  auto dataH = dataT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();

  const dim_t* actIndex = outT->dims().data();
  const dim_t* dstPitch = outT->strides().data();
  void* dst = outT->getRawDataPointer();
  auto numElemsDst = dstPitch[0] * actIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<elkType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dst);
  if (maxRead == 0)
    return;

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

  float invRszBlScale_1 = 0.0;
  float invRszBlScale_2 = 0.0;
  fpReciprocalSingleElement(rszBlScale[1], invRszBlScale_1 );
  fpReciprocalSingleElement(rszBlScale[2], invRszBlScale_2);

  dim_t last_col = -1;
  float ihf, iwf;
  uint32_t ih, iw;
  size_t ih0, ih1, iw0, iw1;

  while (!done && (offsetOut < posMax)) {
    dim_t ob = coord[0]; // sample number (batch)
    dim_t oh = coord[1]; // row (height)
    dim_t ow = coord[2]; // col (width)
    dim_t oc = coord[3]; // channel (channels)

    // This only has to be updated when the column changes
    if (ow != last_col) {
      last_col = ow;
      ihf = static_cast<float>(static_cast<uint32_t>(oh)) * invRszBlScale_1;
      iwf = static_cast<float>(static_cast<uint32_t>(ow)) * invRszBlScale_2;

      ih = static_cast<uint32_t>(ihf);
      iw = static_cast<uint32_t>(iwf);

      ih0 = std::min(static_cast<size_t>(ih), dataT->dims()[1] - 1);
      ih1 = std::min(static_cast<size_t>(ih + 1), dataT->dims()[1] - 1);
      iw0 = std::min(static_cast<size_t>(iw), dataT->dims()[2] - 1);
      iw1 = std::min(static_cast<size_t>(iw + 1), dataT->dims()[2] - 1);
    }

    float dst00, dst01, dst10, dst11;
    if (elKind == Float16Ty) {
      convertFp16ToFp32(static_cast<uint16_t>(dataH.at(std::array<size_t, 4>{ob, ih0, iw0, oc})), dst00);
      convertFp16ToFp32(static_cast<uint16_t>(dataH.at(std::array<size_t, 4>{ob, ih0, iw1, oc})), dst01);
      convertFp16ToFp32(static_cast<uint16_t>(dataH.at(std::array<size_t, 4>{ob, ih1, iw0, oc})), dst10);
      convertFp16ToFp32(static_cast<uint16_t>(dataH.at(std::array<size_t, 4>{ob, ih1, iw1, oc})), dst11);
    } else {
      dst00 =
        dequantize<elkType>(dataH.at(std::array<size_t, 4>{ob, ih0, iw0, oc}), dataH.getScale(), dataH.getOffset());
      dst01 =
        dequantize<elkType>(dataH.at(std::array<size_t, 4>{ob, ih0, iw1, oc}), dataH.getScale(), dataH.getOffset());
      dst10 =
        dequantize<elkType>(dataH.at(std::array<size_t, 4>{ob, ih1, iw0, oc}), dataH.getScale(), dataH.getOffset());
      dst11 =
        dequantize<elkType>(dataH.at(std::array<size_t, 4>{ob, ih1, iw1, oc}), dataH.getScale(), dataH.getOffset());
    }
    float hd = dst00 + (dst10 - dst00) * (ihf - static_cast<float>(ih));
    float hw = dst01 + (dst11 - dst01) * (ihf - static_cast<float>(ih));
    float result = hd + (hw - hd) * (iwf - static_cast<float>(iw));

    if (elKind == Float16Ty) {
      uint16_t out16 = 0;
      convertFp32ToFp16(result, out16);
      outH.at(std::array<size_t, 4>{ob, oh, ow, oc}) = out16;
    } else {
      outH.at(std::array<size_t, 4>{ob, oh, ow, oc}) = quantize<elkType>(result, outT->getScale(), outT->getOffset());
    }

    done = getOffsets(srcDimNum, coord, offsetOut, actIndex, dstPitch);
  }
}

template <ElemKind elKind, size_t N>
INLINE_ATTR
  typename std::enable_if_t<(!isQuantizedElemKind(elKind) && (elKind != Float16Ty) && (elKind != BoolTy)), void>
  fwdLibResizeBilinearInst(LibTensor* outT, LibTensor* dataT, const std::array<float, N>& rszBlScale,
                           [[maybe_unused]] uint64_t flags, const uint32_t minionOffset = 0,
                           [[maybe_unused]] const uint32_t assignedMinions = 0) {

  using elkType = typename elemKind2elemTy<elKind>::type;

  et_assert(get_minion_id() >= minionOffset);

  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  assert(dataT->getElementType() == outT->getElementType());

  /* Scaling batch and channel not supported. */
  assert(rszBlScale[0] == 1.0f);
  assert(rszBlScale[3] == 1.0f);

  auto dataH = dataT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();

  const dim_t* actIndex = outT->dims().data();
  const dim_t* dstPitch = outT->strides().data();
  void* dst = outT->getRawDataPointer();
  auto numElemsDst = dstPitch[0] * actIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<elkType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dst);
  if (maxRead == 0)
    return;

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

  float invRszBlScale_1 = 0.0;
  float invRszBlScale_2 = 0.0;
  fpReciprocalSingleElement(rszBlScale[1], invRszBlScale_1 );
  fpReciprocalSingleElement(rszBlScale[2], invRszBlScale_2);

  dim_t last_col = -1;
  float ihf, iwf;
  uint32_t ih, iw;
  size_t ih0, ih1, iw0, iw1;

  while (!done && (offsetOut < posMax)) {
    size_t ob = coord[0]; // sample number (batch)
    size_t oh = coord[1]; // row (height)
    size_t ow = coord[2]; // col (width)
    size_t oc = coord[3]; // channel (channels)

    // This only has to be updated when the column changes
    if (ow != last_col) {
      last_col = ow;
      ihf = static_cast<float>(static_cast<uint32_t>(oh)) * invRszBlScale_1;
      iwf = static_cast<float>(static_cast<uint32_t>(ow)) * invRszBlScale_2;

      ih = static_cast<uint32_t>(ihf);
      iw = static_cast<uint32_t>(iwf);

      ih0 = std::min(static_cast<size_t>(ih), dataT->dims()[1] - 1);
      ih1 = std::min(static_cast<size_t>(ih + 1), dataT->dims()[1] - 1);
      iw0 = std::min(static_cast<size_t>(iw), dataT->dims()[2] - 1);
      iw1 = std::min(static_cast<size_t>(iw + 1), dataT->dims()[2] - 1);
    }

    auto v00 = dataH.at(std::array<size_t, 4>{ob, ih0, iw0, oc});
    auto v01 = dataH.at(std::array<size_t, 4>{ob, ih0, iw1, oc});
    auto v10 = dataH.at(std::array<size_t, 4>{ob, ih1, iw0, oc});
    auto v11 = dataH.at(std::array<size_t, 4>{ob, ih1, iw1, oc});

    auto hd = static_cast<float>(v00) + static_cast<float>(v10 - v00) * (ihf - static_cast<float>(ih));
    auto hw = static_cast<float>(v01) + static_cast<float>(v11 - v01) * (ihf - static_cast<float>(ih));
    float result = hd + (hw - hd) * (iwf - static_cast<float>(iw));
    if (elKind == BFloat16Ty || elKind == Float16Ty) {
      outH.at(std::array<size_t, 4>{ob, oh, ow, oc}) = static_cast<float>(result);
    } else if (elKind == Int64ITy) {
      outH.at(std::array<size_t, 4>{ob, oh, ow, oc}) = static_cast<uint32_t>(result);
    } else {
      outH.at(std::array<size_t, 4>{ob, oh, ow, oc}) = static_cast<elkType>(result);
    }

    done = getOffsets(srcDimNum, coord, offsetOut, actIndex, dstPitch);
  }
}

} // inlining
} // dnn_lib

#endif
