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
                        const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

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

/**
 * @brief Resizes Generate an Output tensor which is the original one but upscaled
 * by rszScale (if these are natural numbers). The width_scale and height_scale
 * arguments control the size of the output, which is given by:
 * output_width = input_width * width_scale
 * output_height = output_height * height_scale
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
fwdLibResizeNearestInstUpscaleDouble(LibTensor* outT, LibTensor* inT, const std::array<float, N>& rszScale,
                                     uint64_t flags, const uint32_t minionOffset = 0,
                                     const uint32_t assignedMinions = 0) {
  using elkType = typename elemKind2elemTy<elKind>::type;

  et_assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  et_assert(inT->getElementType() == outT->getElementType());

  constexpr size_t typeSize = getsize<elkType>();

  elkType* dst = (elkType*)(outT->getRawDataPointer());
  elkType* src = (elkType*)(inT->getRawDataPointer());

  const dim_t nDims = inT->ndims();
  const dim_t* srcDims = inT->dims().data();
  const dim_t* dstDims = outT->dims().data();
  const dim_t* srcStride = inT->strides().data();
  const dim_t* dstStride = outT->strides().data();

  auto numElemsDst = dstStride[0] * dstDims[0];

  size_t dstOffsetElms, maxRead;
  getCachelinePartition(typeSize, numElemsDst, dstOffsetElms, maxRead, minionId, activeMinions, dst);
  if (maxRead == 0)
    return;
  size_t dstOffsetElmsMax = dstOffsetElms + maxRead;

  dim_array_t intRszScale = {0};
  for (size_t i = 0; i < N; i++) {
    intRszScale[i] = static_cast<uint32_t>(rszScale[i]);
  }

  // Move dstOffsetElms to the next non-padding position in dst
  dim_array_t coordDst = {0}; // Vector of coordinates
  dim_t k = 0;                // Amount of non-zero coordinates
  getNonPaddingCoordinates(coordDst, dstOffsetElms, nDims, dstStride, dstDims, k);

  dstOffsetElms = 0;
  size_t srcOffsetElms = 0;
  dim_array_t coordSrc = {0};
  for (size_t i = 0; i < k; i++) {
    coordSrc[i] = coordDst[i] / intRszScale[i];
    dstOffsetElms += coordDst[i] * dstStride[i];
    srcOffsetElms += coordSrc[i] * srcStride[i];
  }

  bool done = false;
  constexpr size_t m0Masks[] = {0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff};
  while (not done and (dstOffsetElms < dstOffsetElmsMax)) {
    // Calc how many values can be read and advance that number in coordinates
    int nValues = std::min(srcDims[nDims - 1] - coordSrc[nDims - 1], 8UL);
    elkType* realSrcAddr = src + srcOffsetElms;
    elkType* realDstAddr = dst + dstOffsetElms;
    if (typeSize == 2) {
      constexpr size_t rs1Size2Read0 = 0x76543210; // 0b0111 0110 0101 0100 0011 0010 0001 0000
      constexpr size_t rs1Size2Read1 = 0xFEDCBA98; // 0b1111 1110 1101 1100 1011 1010 1001 1000
      // low if address%32 == 0, otherwise address%32 == 16
      bool load_low = ((uintptr_t(realSrcAddr) & 0x1f) == 0);
      elkType* alignedSrcAddr = (elkType*)((uintptr_t(realSrcAddr) >> 5) << 5); // align to 32 byte
      size_t m0Value = m0Masks[nValues];
      // read 16 bytes from origin tensor and write 32 bytes to dest tensor
      __asm__ __volatile__(
        "mov.m.x   m0, %[m0Value], 0x00\n"          // load the mask to m0
        "fg32h.ps  f0, %[rs1Load](%[srcAddr])\n"    // load the data to f0
        "fsc32h.ps f0, %[rs1Write0](%[dstAddr0])\n" // store data in even positions -> 0 X 1 X 2 X 3 X 4 X 5 X 6 X 7 X
        "fsc32h.ps f0, %[rs1Write1](%[dstAddr0])\n" // store data in odd positions  -> 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7
        :
        : [ rs1Load ] "r"(load_low ? rs1Size2Read0 : rs1Size2Read1), [ srcAddr ] "r"(alignedSrcAddr),
          [ dstAddr0 ] "r"(realDstAddr), [ m0Value ] "r"(m0Value),
          // Literal for writting values in even possitions
          // 0xECA86420 = 0b1110 1100 1010 1000 0110 0100 0010 0000
          [ rs1Write0 ] "r"(0xECA86420),
          // Literal for even positions
          // 0xFDB97531 = 0b1111 1101 1011 1001 0111 0101 0011 0001
          [ rs1Write1 ] "r"(0xFDB97531)
        : "f0");
    } else if (typeSize == 4) {
      for (int i = 0; i < 2; i++) {
        constexpr size_t rs1Size4Read = 0x00FAC688; // 0b111 110 101 100 011 010 001 000
        // Each one of the two iterations of the loop moves 4 elms, so if there are
        // less than 5 elms to move the second iteration is not needed
        if (not(nValues < 5 and i == 1)) {
          bool load_low = (i == 0);
          [[maybe_unused]] size_t m0Value;
          if (i == 0) {
            m0Value = m0Masks[std::min(4, nValues)];
          } else {
            m0Value = m0Masks[nValues - 4];
          }
          // read 16 bytes (4 elements) from origin tensor and write 32 bytes to dest tensor
          __asm__ __volatile__(
            "mov.m.x   m0, %[m0Value], 0x00\n"       // write the mask to m0
            "fg32w.ps  f0, %[rs1Load](%[srcAddr])\n" // load the data to f0
            "li        x31, 0x0D10\n"                // 110 100 010 000, values to be writen in even positions
            "fsc32w.ps f0, x31(%[dstAddr0])\n"       // store data in even positions -> 0 X 1 X 2 X 3 X
            "li        x31, 0x0F59\n"                // 111 101 011 001, values to be writen in odd positions
            "fsc32w.ps f0, x31(%[dstAddr0])\n"       // store data in even positions -> 0 0 1 1 2 2 3 3
            :
            : [ rs1Load ] "r"(load_low ? rs1Size4Read : rs1Size4Read >> 3 * 4),    // 3 bits per elm, 4 elms
              [ srcAddr ] "r"(realSrcAddr), [ dstAddr0 ] "r"(realDstAddr + 8 * i), // only offset 8 elms when i == 1
              [ m0Value ] "r"(m0Value)
            : "f0", "x31");
        }
      }
    } else {
      et_assert(typeSize == 2 or typeSize == 4);
    }
    coordSrc[nDims - 1] += nValues;
    srcOffsetElms += nValues;
    coordDst[nDims - 1] += nValues * 2;
    dstOffsetElms += nValues * 2;
    // Check for end of row
    if (coordSrc[nDims - 1] >= srcDims[nDims - 1]) {
      et_assert(coordSrc[nDims - 1] == srcDims[nDims - 1])
        // Jump padding and detect end of tensor
        for (size_t dim = nDims - 1; dim >= 0; dim--) {
        if (unlikely(coordDst[dim] == dstDims[dim])) {
          coordDst[dim] = 0;
          done = (dim == 0);
        } else {
          coordDst[dim]++;
          break;
        }
      }
      // Update offset elms and src coords
      if (likely(not done)) {
        srcOffsetElms = 0;
        dstOffsetElms = 0;
        for (size_t i = 0; i < k; i++) {
          coordSrc[i] = coordDst[i] / intRszScale[i];
          srcOffsetElms += coordSrc[i] * srcStride[i];
          dstOffsetElms += coordDst[i] * dstStride[i];
        }
      }
    }
  }
}

}  // inlining
}  // dnn_space

#endif
