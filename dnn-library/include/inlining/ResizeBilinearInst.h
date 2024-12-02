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

/**
 * @brief Resize the input tensor by (1,1,2,2) using linear interpolation.
 * Supported ElemKind: FloatTy, Float16Ty, Int8QTy, Int16QTy, Int32QTy, Int32ITy,
 *                     Int64ITy
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected.
 * @param[in] inT LibTensor input. It keeps the inputs to being handle.
 * @param[in] rszBlScale This array keeps the scale to be applied at each dimension.
 * @param[flags] flags Gives the information of the Active Shires and the
 * type of evict required.
 */
template <ElemKind elKind, size_t N>
INLINE_ATTR typename std::enable_if_t<elKind == Float16Ty, void>
fwdLibResizeBilinearInstUpscaleDouble(LibTensor* outT, LibTensor* inT, const std::array<float, N>& rszBlScale,
                                      uint64_t flags, const uint32_t minionOffset = 0,
                                      const uint32_t assignedMinions = 0) {
  using elkType = typename elemKind2elemTy<elKind>::type;
  et_assert(get_minion_id() >= minionOffset);

  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  assert(inT->getElementType() == outT->getElementType());

  /* Scaling batch and channel not supported. */
  assert(rszBlScale[0] == 1.0f);
  assert(rszBlScale[3] == 1.0f);

  elkType* dst_ptr = (elkType*)(outT->getRawDataPointer());
  elkType* src_ptr = (elkType*)(inT->getRawDataPointer());

  const int nDims = outT->ndims();
  const dim_t* dstDims = outT->dims().data();
  const dim_t* dstStride = outT->strides().data();
  const dim_t* srcStride = inT->strides().data();
  auto numElemsDst = dstStride[0] * dstDims[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<elkType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dst_ptr);
  if (maxRead == 0)
    return;

  /*
   * When upscaling a 2x2 tensor using interpolation there are four types of cells:
   *
   *  0. Those that coincide with a cell in the original tensor and don't need
   *     interpolation. For example dst location (0,2,2,0) gets its value directly
   *     from the item in src at position (0,1,1,0).
   *  1. Those that coincide with an even row (starting from 0) but with an odd
   *     column need interpolation between two columns from the original tensor.
   *     For example dst location (0,2,3,0) gets its value from the interpolation
   *     of src values at (0,1,1,0) and (0,1,2,0).
   *  2. Those that coincide with odd row and even column need interpolation
   *     between two rows in the src tensor. For example dst location (0,3,2,0)
   *     gets its value from the interpolation of src locations (0,1,1,0) and
   *     (0,2,1,0).
   *  3. Those that coincide with both odd row and column need to interpolate both
   *     row and column. For example, the dst location (0,3,3,0) gets its value
   *     from the interpolation of src items at (0,1,1,0), (0,1,2,0), (0,2,1,0)
   *     and (0,2,2,0).
   *
   * As the tensor are organized in (b,h,w,c), and we only interpolate h and w,
   * it was found that if each minion gets a contiguous chunk of the dst tensor
   * minions that do the channels (c) that correspond to interpolations of type
   * 0 have to do almost no work, whereas minions that do 1 or 2 require almost
   * double the work with respect to type 0, and those that have to do interpolation
   * of type 3 do double as much work as those who do 1 or 2. This imbalance is
   * not ideal and, in order to solve it, we have to make each minion do the four
   * types of interpolations if possible. Thus, if originally a minion would do
   * a channel of type 0, another a channel of type 1, and so on for 2 and 3, now
   * the first minion does the first 1/4 of types 0, 1, 2 and 3, the second minion
   * does the second 1/4 of types 0, 1, 2 and 3, and so on.
   */

  dim_array_t initialAddrChunks = {0, 0, 0, initialAddr};
  dim_t chunkSize = maxRead;
  dim_t currentChunk = 3;
  // Channel dimension has enough cache lines so that its work can be split in 4
  if (((((maxRead * typeSize) / CACHE_LINE_BYTES) % 4) == 0) and (maxRead <= dstStride[2])) {
    currentChunk = 0;
    chunkSize /= 4;
    dim_t origChunkType = ((((initialAddr % dstStride[1]) / dstStride[2]) % 2) != 0) +
                          2 * ((((initialAddr % dstStride[0]) / dstStride[1]) % 2) != 0);
    dim_t rowOffset = ((initialAddr / dstStride[1]) % 2) * dstStride[1];
    dim_t colOffset = (((initialAddr % dstStride[1]) / dstStride[2]) % 2) * dstStride[2];
    dim_t partitionBaseAddr = initialAddr - rowOffset - colOffset + chunkSize * origChunkType;
    initialAddrChunks[0] = partitionBaseAddr;
    initialAddrChunks[1] = partitionBaseAddr + dstStride[2];
    initialAddrChunks[2] = partitionBaseAddr + dstStride[1];
    initialAddrChunks[3] = partitionBaseAddr + dstStride[2] + dstStride[1];
    // If the channel dimension has less than four cache lines then the work is split
    // in two instead of four. If the work cannot be split in two (very small channel dimension),
    // then the work inbalance can't be fix.
  } else if ((maxRead <= dstStride[1]) and (maxRead > dstStride[2]) and ((maxRead % dstStride[2]) == 0)) {
    currentChunk = 2;
    chunkSize /= 2;
    [[maybe_unused]] dim_t origChunkType = ((((initialAddr % dstStride[0]) / dstStride[1]) % 2) != 0);
    [[maybe_unused]] dim_t rowOffset = ((initialAddr / dstStride[1]) % 2) * dstStride[1];
    [[maybe_unused]] dim_t partitionBaseAddr = initialAddr - rowOffset + chunkSize * origChunkType;
    initialAddrChunks[2] = partitionBaseAddr;
    initialAddrChunks[3] = partitionBaseAddr + dstStride[1];
  }
  initialAddr = initialAddrChunks[currentChunk];

  // We move the initialAddr to the next non-padding position
  dim_array_t coordDst = {0}; // Vector of output coordinates
  dim_t k = 0;                // Amount of non-zero coordinates
  dim_t srcDimNum = inT->ndims();
  getNonPaddingCoordinates(coordDst, initialAddr, srcDimNum, dstStride, dstDims, k);

  dim_array_t coordSrc = {coordDst[0], coordDst[1] / 2, coordDst[2] / 2, coordDst[3]}; // Vector of input coordinates
  uint64_t srcOffsetElms = 0;
  uint64_t dstOffsetElms = 0;
  for (dim_t j = 0; j < k; j++) {
    srcOffsetElms += srcStride[j] * coordSrc[j];
    dstOffsetElms += dstStride[j] * coordDst[j];
  }

  bool done = false;
  size_t totalProc = dstOffsetElms - initialAddr;
  constexpr size_t m0Masks[] = {0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff};
  constexpr size_t rs1 = 0x76543210; // 0b0111 0110 0101 0100 0011 0010 0001 0000
  while (not done) {
    dim_t to_process = std::min(8ul, dstDims[3] - coordDst[3]); // process up to 8 elements or end of row
    totalProc += to_process;
    elkType* realSrcAddr = src_ptr + srcOffsetElms;
    elkType* realDstAddr = dst_ptr + dstOffsetElms;
    auto col_pp = (unlikely(coordDst[2] == dstDims[2] - 1) ? 0 : srcStride[2]);
    auto row_pp = (unlikely(coordDst[1] == dstDims[1] - 1) ? 0 : srcStride[1]);

    if (coordDst[1] % 2 == 0 and coordDst[2] % 2 == 0) {
      // Channel values do not need to be interpolated, they coincide with the original dims
      __asm__ __volatile__("mov.m.x   m0, %[m0Value], 0x00\n"     // load the mask to m0
                           "fg32h.ps  f0, %[rs1](%[srcAddr00])\n" // load the data to f0
                           "fsc32h.ps f0, %[rs1](%[dstAddr])\n"   // store data from f0 to dst
                           :
                           : [ rs1 ] "r"(rs1), [ srcAddr00 ] "r"(realSrcAddr), [ dstAddr ] "r"(realDstAddr),
                             [ m0Value ] "r"(m0Masks[to_process])
                           : "f0");
    } else if (coordDst[1] % 2 == 1 and coordDst[2] % 2 == 1) {
      // Channel values need row and column-wise interpolation, that is,
      // (srcAddr00 + srcAddr01 + srcAddr10 + srcAddr11)*0.25
      __asm__ __volatile__("mov.m.x   m0, %[m0Value], 0x00\n"     // load the mask to m0
                           "fg32h.ps  f0, %[rs1](%[srcAddr00])\n" // load the data at dst00 to f0
                           "fg32h.ps  f1, %[rs1](%[srcAddr01])\n"
                           "fcvt.ps.f16 f0, f0\n" // convert f0 from fp16 to fp32
                           "fcvt.ps.f16 f1, f1\n"
                           "fadd.ps   f0, f0, f1, dyn\n" // f0 = dst00 + dst01
                           "fg32h.ps  f1, %[rs1](%[srcAddr10])\n"
                           "fcvt.ps.f16 f1, f1\n"
                           "fadd.ps   f0, f0, f1, dyn\n" // f0 = dst00 + dst01 + dst10
                           "fg32h.ps  f1, %[rs1](%[srcAddr11])\n"
                           "fcvt.ps.f16 f1, f1\n"
                           "fadd.ps   f0, f0, f1, dyn\n"        // f0 = dst00 + dst01 + dst10 + dst11
                           "fbci.ps   f1, 0x3e800\n"            // load 0.25 into all values of f1
                           "fmul.ps   f0, f0, f1\n"             // divide by four to get the avg in f0
                           "fcvt.f16.ps f0, f0\n"               // convert f0 from fp32 back to fp16
                           "fsc32h.ps f0, %[rs1](%[dstAddr])\n" // store interpolated data to dst
                           :
                           : [ rs1 ] "r"(rs1), [ srcAddr00 ] "r"(realSrcAddr), [ srcAddr01 ] "r"(realSrcAddr + col_pp),
                             [ srcAddr10 ] "r"(realSrcAddr + row_pp), [ srcAddr11 ] "r"(realSrcAddr + row_pp + col_pp),
                             [ dstAddr ] "r"(realDstAddr), [ m0Value ] "r"(m0Masks[to_process])
                           : "f0", "f1");
    } else {
      // Channel values need row or column-wise interpolation, that is,
      // (srcAddr00 + srcAddr10)*0.5 or (srcAddr00 + srcAddr01)*0.5
      auto offset = (coordDst[1] % 2 == 0 ? col_pp : row_pp);
      __asm__ __volatile__("mov.m.x   m0, %[m0Value], 0x00\n"       // load the mask to m0
                           "fg32h.ps  f0, %[rs1](%[srcAddr00])\n"   // load the data at dst00 to f0
                           "fg32h.ps  f1, %[rs1](%[srcAddrDiag])\n" // load the data at dstDiag to f1
                           "fcvt.ps.f16 f0, f0\n"                   // convert f0 from fp16 to fp32
                           "fcvt.ps.f16 f1, f1\n"                   // convert f1 from fp16 to fp32
                           "fadd.ps   f0, f0, f1, dyn\n"            // add f0 + f1 and store in f0
                           "fbci.ps   f1, 0x3f000\n"                // load 0.5 into all values of f1
                           "fmul.ps   f0, f0, f1\n"                 // divide by two to get the avg in f0
                           "fcvt.f16.ps f0, f0\n"                   // convert f0 from fp32 back to fp16
                           "fsc32h.ps f0, %[rs1](%[dstAddr])\n"     // store interpolated data to dst
                           :
                           : [ rs1 ] "r"(rs1), [ srcAddr00 ] "r"(realSrcAddr),
                             [ srcAddrDiag ] "r"(realSrcAddr + offset), [ dstAddr ] "r"(realDstAddr),
                             [ m0Value ] "r"(m0Masks[to_process])
                           : "f0", "f1");
    }

    coordSrc[nDims - 1] += to_process;
    srcOffsetElms += to_process;
    coordDst[nDims - 1] += to_process;
    dstOffsetElms += to_process;
    bool updateSrc = false;
    // Check for end of row
    if (unlikely(coordDst[nDims - 1] >= dstDims[nDims - 1])) {
      et_assert(coordDst[nDims - 1] == dstDims[nDims - 1])
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
      // Update offset elms and coords
      if (likely(not done)) {
        updateSrc = true;
        dstOffsetElms = 0;
        for (size_t i = 0; i < k; i++) {
          dstOffsetElms += coordDst[i] * dstStride[i];
        }
      }
    }
    // Check for end of chunk (need to jump to the next type of interpolation)
    if (unlikely(dstOffsetElms >= initialAddrChunks[currentChunk] + chunkSize)) {
      et_assert(dstOffsetElms == initialAddrChunks[currentChunk] + chunkSize);
      if (currentChunk == 3) {
        done = true;
      } else {
        currentChunk++;
        dstOffsetElms = initialAddrChunks[currentChunk];
        coordDst[0] = dstOffsetElms / dstStride[0];
        coordDst[1] = (dstOffsetElms % dstStride[0]) / dstStride[1];
        coordDst[2] = (dstOffsetElms % dstStride[1]) / dstStride[2];
        coordDst[3] = (dstOffsetElms % dstStride[2]) / dstStride[3];
        updateSrc = true;
      }
    }
    if (updateSrc) {
      updateSrc = false;
      coordSrc[0] = coordDst[0];
      coordSrc[1] = coordDst[1] / 2;
      coordSrc[2] = coordDst[2] / 2;
      coordSrc[3] = coordDst[3];
      srcOffsetElms = 0;
      for (size_t i = 0; i < k; i++) {
        srcOffsetElms += coordSrc[i] * srcStride[i];
      }
    }
  }
}

template <ElemKind elKind, size_t N>
INLINE_ATTR typename std::enable_if_t<elKind != Float16Ty, void>
fwdLibResizeBilinearInstUpscaleDouble([[maybe_unused]] LibTensor* outT, [[maybe_unused]] LibTensor* dataT,
                                      [[maybe_unused]] const std::array<float, N>& rszBlScale,
                                      [[maybe_unused]] uint64_t flags, [[maybe_unused]] const uint32_t minionOffset = 0,
                                      [[maybe_unused]] const uint32_t assignedMinions = 0) {
}
} // inlining
} // dnn_lib

#endif
