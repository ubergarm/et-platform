/*-------------------------------------------------------------------------
 * Copyright (C) 2024, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _DEQUANTIZE_4BITS_BLOCKS_INST_H_
#define _DEQUANTIZE_4BITS_BLOCKS_INST_H_

#include "Addresser.h"
#include "Float16.h"
#include "LibTensor.h"
#include "etsoc/common/utils.h"
#include "utils.h"
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

/**
 * @brief Dequantize a matrix of 4-bit elements with quantization parameters
 * defined per blocks inside columns.
 *
 * The usual dequantization formula is applied to each element of the matrix:
 * $dequantized = (quantized - offset) * scale$.
 * Unlike in standard dequantize, both the quantized values and the offsets are
 * expressed as uint4 (their difference must be computed in a larger signed
 * type). Each byte contains a pair of uint4 elements (the first uint4 is the
 * least significant half, the second uint4 is the most significant half).
 * This is the single-threaded version of the operation.
 *
 * @tparam dstElK The floating-point type of the elements in the output tensor.
 * @param[out] outT LibTensor pointer to the output matrix.
 * @param[in] inT LibTensor pointer to the input matrix.
 * @param[in] scaleT LibTensor pointer to the vector of scales.
 * @param[in] offsetT LibTensor pointer to the vector of offsets.
 * @param[in] flags Controls the active shires and the type of evict that
 * should be done at the end of the function.
 */
template <ElemKind dstElK>
INLINE_ATTR void fwdLibDequantize4BitsColumnBlocksInst(LibTensor* outT, LibTensor* inT, LibTensor* scaleT,
                                                       LibTensor* offsetT, uint64_t flags,
                                                       const uint32_t minionOffset = 0,
                                                       [[maybe_unused]] const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) {
    return;
  }

  using fpType = typename elemKind2elemTy<dstElK>::type;
  et_assert(inT->getElementType() == UInt8QTy);
  et_assert(offsetT->getElementType() == UInt8QTy);
  et_assert(scaleT->getElementType() == outT->getElementType());

  // Tensor handles:
  auto outH = outT->getHandle<fpType>();
  auto inH = inT->getHandle<uint8_t>();
  auto scaleH = scaleT->getHandle<fpType>();
  auto offsetH = offsetT->getHandle<uint8_t>();

  // Dimension sizes:
  et_assert(outT->ndims() == 2);
  dim_t numRows = outT->dims()[0];
  dim_t numCols = outT->dims()[1];
  dim_t numBlocksPerCol = scaleT->dims()[0] / numCols;
  // numElemsPerBlock = [ first power of 2 that is >= numRows/numBlocksPerCol ].
  dim_t numElemsPerBlock = (numRows + numBlocksPerCol - 1) / numBlocksPerCol;
  dim_t nextPowerOf2 = 1;
  while (nextPowerOf2 < numElemsPerBlock) {
    nextPowerOf2 *= 2;
  }
  numElemsPerBlock = nextPowerOf2;

  // Interleave:
  dim_array_t inStrides = inT->strides();
  dim_array_t outStrides = outT->strides();
  dim_t interleaveFactor = outStrides[1];
  et_assert(inStrides[1] == interleaveFactor);
  et_assert(numElemsPerBlock % interleaveFactor == 0);
  numRows = (numRows + interleaveFactor - 1) / interleaveFactor;
  numCols *= interleaveFactor;
  inStrides[1] /= interleaveFactor;
  outStrides[1] /= interleaveFactor;

  for (dim_t dstRow = 0; dstRow < numRows; ++dstRow) {
    for (dim_t dstCol = 0; dstCol < numCols; ++dstCol) {
      // Compute coordinates in tensors.
      std::array<dim_t, 2> outCoord = {dstRow, dstCol};
      std::array<dim_t, 2> inByteCoord = {dstRow, dstCol / 2};
      dim_t blockIdx =
        (numBlocksPerCol * (dstCol / interleaveFactor)) + ((dstRow * interleaveFactor) / numElemsPerBlock);
      std::array<dim_t, 1> scaleCoord = {blockIdx};
      std::array<dim_t, 1> offsetByteCoord = {blockIdx / 2};
      // Extract operands from the tensors.
      uint8_t quantizedPack = inH.at(inByteCoord, inStrides, 1);
      float quantizedElement;
      if (dstCol % 2 == 0) {
        // Lower half of the byte.
        quantizedElement = static_cast<float>(quantizedPack & 0x0F);
      } else {
        // Higher half of the byte.
        quantizedElement = static_cast<float>(quantizedPack >> 4);
      }
      uint8_t offsetPack = offsetH.at(offsetByteCoord);
      float offset;
      if (blockIdx % 2 == 0) {
        // Lower half of the byte.
        offset = static_cast<float>(offsetPack & 0x0F);
      } else {
        // Higher half of the byte.
        offset = static_cast<float>(offsetPack >> 4);
      }
      float scale;
      if constexpr (dstElK == Float16Ty) {
        convertFp16ToFp32(scaleH.at(scaleCoord), scale);
      } else {
        static_assert(dstElK == FloatTy);
        scale = scaleH.at(scaleCoord);
      }
      // Compute dequantization: all intermediate computations in float.
      float dequantizedElement = (quantizedElement - offset) * scale;
      if constexpr (dstElK == Float16Ty) {
        convertFp32ToFp16(dequantizedElement, outH.at(outCoord, outStrides, 1));
      } else {
        static_assert(dstElK == FloatTy);
        outH.at(outCoord, outStrides, 1) = dequantizedElement;
      }
    }
  }

  outT->evict(DO_EVICTS);
}

/**
 * @brief Dequantize a matrix of 4-bit elements with quantization parameters
 * defined per blocks inside columns.
 *
 * The usual dequantization formula is applied to each element of the matrix:
 * $dequantized = (quantized - offset) * scale$.
 * Unlike in standard dequantize, both the quantized values and the offsets are
 * expressed as uint4 (their difference must be computed in a larger signed
 * type). Each byte contains a pair of uint4 elements (the first uint4 is the
 * least significant half, the second uint4 is the most significant half).
 * This is the threaded version of the operation.
 *
 * @tparam dstElK The floating-point type of the elements in the output tensor.
 * @param[out] outT LibTensor pointer to the output matrix.
 * @param[in] inT LibTensor pointer to the input matrix.
 * @param[in] scaleT LibTensor pointer to the vector of scales.
 * @param[in] offsetT LibTensor pointer to the vector of offsets.
 * @param[in] flags Controls the active shires and the type of evict that
 * should be done at the end of the function.
 */
template <ElemKind dstElK>
INLINE_ATTR void fwdLibDequantize4BitsColumnBlocksInstThreaded(LibTensor* outT, LibTensor* inT, LibTensor* scaleT,
                                                               LibTensor* offsetT, uint64_t flags,
                                                               const uint32_t minionOffset = 0,
                                                               const uint32_t assignedMinions = 0) {

  et_assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;

  if (minionId >= activeMinions) {
    return;
  }

  using fpType = typename elemKind2elemTy<dstElK>::type;
  et_assert(inT->getElementType() == UInt8QTy);
  et_assert(offsetT->getElementType() == UInt8QTy);
  et_assert(scaleT->getElementType() == outT->getElementType());

  // Tensor handles:
  auto outH = outT->getHandle<fpType>();
  auto inH = inT->getHandle<uint8_t>();
  auto scaleH = scaleT->getHandle<fpType>();
  auto offsetH = offsetT->getHandle<uint8_t>();

  // Dimension sizes:
  et_assert(outT->ndims() == 2);
  dim_t numRows = outT->dims()[0];
  dim_t numCols = outT->dims()[1];
  dim_t numBlocksPerCol = scaleT->dims()[0] / numCols;
  // numElemsPerBlock = [ first power of 2 that is >= numRows/numBlocksPerCol ].
  dim_t numElemsPerBlock = (numRows + numBlocksPerCol - 1) / numBlocksPerCol;
  dim_t nextPowerOf2 = 1;
  while (nextPowerOf2 < numElemsPerBlock) {
    nextPowerOf2 *= 2;
  }
  numElemsPerBlock = nextPowerOf2;

  // Interleave:
  dim_array_t inStrides = inT->strides();
  dim_array_t outStrides = outT->strides();
  dim_array_t outDims = outT->dims();
  dim_t interleaveFactor = outStrides[1];
  et_assert(inStrides[1] == interleaveFactor);
  et_assert(numElemsPerBlock % interleaveFactor == 0);
  numRows = (numRows + interleaveFactor - 1) / interleaveFactor;
  numCols *= interleaveFactor;
  outDims[0] = numRows;
  outDims[1] = numCols;
  inStrides[1] /= interleaveFactor;
  outStrides[1] /= interleaveFactor;

  // Raw parameters for the output tensor:
  void* dstT = outT->getRawDataPointer();
  const dim_t* dstPitch = outStrides.data();
  const dim_t* dstDims = outDims.data();
  size_t numDims = 2;
  size_t numElemsDst = dstPitch[0] * numRows; // Total number of elements in the tensor

  // We give to each minion an initial address and the number of positions that
  // it must work on (maxRead).
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<fpType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dstT);

  if (maxRead == 0) {
    return;
  }

  // We move the initialAddr to the next non-padding position
  dim_array_t coord = {0}; // Vector of coordinates
  size_t k = 0;            // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, numDims, dstPitch, dstDims, k);

  // We get the actual initialAddr, in the output.
  size_t offsetOut = 0;
  for (size_t j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }

  // In each iteration we dequantize a position and switch to the next one,
  // until completion.
  auto posMax = maxRead + initialAddr;
  bool done = false;
  while (not done and (offsetOut < posMax)) {
    // Compute coordinates in tensors.
    dim_t dstRow = coord[0];
    dim_t dstCol = coord[1];
    std::array<dim_t, 2> outCoord = {dstRow, dstCol};
    std::array<dim_t, 2> inByteCoord = {dstRow, dstCol / 2};
    dim_t blockIdx = (numBlocksPerCol * (dstCol / interleaveFactor)) + ((dstRow * interleaveFactor) / numElemsPerBlock);
    std::array<dim_t, 1> scaleCoord = {blockIdx};
    std::array<dim_t, 1> offsetByteCoord = {blockIdx / 2};
    // Extract operands from the tensors.
    uint8_t quantizedPack = inH.at(inByteCoord, inStrides, 1);
    float quantizedElement;
    if (dstCol % 2 == 0) {
      // Lower half of the byte.
      quantizedElement = static_cast<float>(quantizedPack & 0x0F);
    } else {
      // Higher half of the byte.
      quantizedElement = static_cast<float>(quantizedPack >> 4);
    }
    uint8_t offsetPack = offsetH.at(offsetByteCoord);
    float offset;
    if (blockIdx % 2 == 0) {
      // Lower half of the byte.
      offset = static_cast<float>(offsetPack & 0x0F);
    } else {
      // Higher half of the byte.
      offset = static_cast<float>(offsetPack >> 4);
    }
    float scale;
    if constexpr (dstElK == Float16Ty) {
      convertFp16ToFp32(scaleH.at(scaleCoord), scale);
    } else {
      static_assert(dstElK == FloatTy);
      scale = scaleH.at(scaleCoord);
    }
    // Compute dequantization: all intermediate computations in float.
    float dequantizedElement = (quantizedElement - offset) * scale;
    if constexpr (dstElK == Float16Ty) {
      convertFp32ToFp16(dequantizedElement, outH.at(outCoord, outStrides, 1));
    } else {
      static_assert(dstElK == FloatTy);
      outH.at(outCoord, outStrides, 1) = dequantizedElement;
    }
    // Prepare next iteration (if any).
    done = getOffsets(numDims, coord, offsetOut, dstDims, dstPitch) or (offsetOut >= posMax);
  }

  if (not DO_EVICTS) {
    return;
  }
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) {
    evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize * initialAddr, clperminion);
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _DEQUANTIZE_4BITS_BLOCKS_INST_H_
