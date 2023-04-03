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

#ifndef _CONVOLUTION_3D_INST_H_
#define _CONVOLUTION_3D_INST_H_

#include "Addresser.h" // From include/internal path
#include "Float16.h"
#include "LibTensor.h"
#include "LoadStore.h" // From include/internal path

#include "utils.h" // From include/internal path
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

/// \brief Computes one output element on the convolution.
///
/// This code is not vectorized.
///
/// \tparam dstElK Element kind of source and destination tensors
/// \tparam biasElK Element kind of bias
/// \tparam N Size of kernel_size array (compiler-provided through automatic
/// template deduction)
/// \param[in] activations Pointer to activation tensor data
/// \param[in] weights Pointer to filter tensor data
/// \param[in] coord The array of coordinates to the initial position in the
///  activations. coord[0] corresponds to the batch and coord[3] corresponds
///  to the group where we are.
/// \param[in] actPitch array of pitches of the activations matrix.
/// \param[in] weightPitch Array of pitches of the weights matrix.
/// \param[in] actIndex Array of the size of each dimensions of the activations.
/// \param[in] kernels Sime of the kernel (array of N elements)
/// \param[in] inCperG Elements in a group of channel from the source tensor.
/// \param[in] x, y, z, d Coordinates where our minions should start reading.
/// \param[in] scale Array of tensor scales
/// \param[in] offset Array of tensor offsets

template <ElemKind dstElK, ElemKind biasElK, size_t N>
INLINE_ATTR void quantConvolution3DOp(void* activations, void* weights, void* bias, void* output, size_t offsetOut,
                                      const dim_array_t& coord, const dim_t* actPitch, const dim_t* weightPitch,
                                      const dim_t* actIndex, const std::array<uint32_t, N>& kernels, dim_t inCperG,
                                      ssize_t x, ssize_t y, ssize_t z, ssize_t d, const float& inScale,
                                      const float& filterScale, const float& biasScale, const float& outScale,
                                      const int32_t& inOffset, const int32_t& filterOffset, const int32_t& biasOffset,
                                      const int32_t& outOffset) {

  using ElemType = typename AccumulatingQuantizedOpTypes<dstElK, biasElK>::elemType;
  using AccumulatorType = typename AccumulatingQuantizedOpTypes<dstElK, biasElK>::accumulatorType;
  using BiasType = typename AccumulatingQuantizedOpTypes<dstElK, biasElK>::biasType;

  ElemType& result = static_cast<ElemType*>(output)[offsetOut];

  // Compute matMulScale and matMulScaleRec
  //
  // float<8> matMulScale = broadcast<8>(inScale) * broadcast<8>(filterScale)
  // float<8> matMulScaleRec = rec(matMulScale)
  float matMulScale, matMulScaleRec;
  float tmp, tmp2;
  __asm__ __volatile__("fbcx.ps %[tmp], %[inScale]\n"
                       "fbcx.ps %[tmp2], %[filterScale]\n"
                       "fmul.ps %[matMulScale], %[tmp], %[tmp2]\n"
                       "frcp.ps %[matMulScaleRec], %[matMulScale]\n"
                       : [ matMulScale ] "=f"(matMulScale), [ tmp ] "=&f"(tmp), [ tmp2 ] "=f"(tmp2),
                         [ matMulScaleRec ] "=f"(matMulScaleRec)
                       : [ inScale ] "r"(inScale), [ filterScale ] "r"(filterScale));

  // Compute output quantization parameters, including matMulScale premultiply:
  //
  // float<8> outQuantScaleRec = rec(broadcast<8>(outScale)) * matMulScale
  // float<8> outQuantOffset = convert<float>(broadcast<8>(outOffset))
  float outQuantScaleRec, outQuantOffset;
  setupQuantize(outQuantScaleRec, outQuantOffset, outScale, outOffset);
  __asm__ __volatile__("fmul.ps %[outQuantScaleRec], %[outQuantScaleRec], %[matMulScale]\n"
                       : [ outQuantScaleRec ] "+f"(outQuantScaleRec)
                       : [ matMulScale ] "f"(matMulScale));

  // Compute B
  //
  const BiasType& biasValue = static_cast<BiasType*>(bias)[d];
  float Bfloat = (static_cast<float>(biasValue) - static_cast<float>(biasOffset)) * biasScale * matMulScaleRec;
  convertFloatToInt32<RoundingMode::LikeStdRoundAndCast>(Bfloat, Bfloat);
  int64_t Bint64;
  __asm__ __volatile__("fmvs.x.ps %[first], %[tmp], 0\n" : [ first ] "=r"(Bint64) : [ tmp ] "f"(Bfloat));
  AccumulatorType B = static_cast<AccumulatorType>(Bint64);

  // Scalar code for weighted sum
  //
  AccumulatorType sum = 0;
  for (size_t fx = 0; fx < kernels[0]; fx++) {
    for (size_t fy = 0; fy < kernels[1]; fy++) {
      for (size_t fz = 0; fz < kernels[2]; fz++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;
        ssize_t oz = z + fz;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || oz < 0 || ox >= ssize_t(actIndex[1]) || oy >= ssize_t(actIndex[2]) ||
            oz >= ssize_t(actIndex[3])) {
          continue;
        }

        for (size_t fd = 0; fd < inCperG; fd++) { // for all depth coordinates
          size_t index1 = d * weightPitch[0] + fx * weightPitch[1] + fy * weightPitch[2] + fz * weightPitch[3] + fd;
          size_t index2 = coord[0] * actPitch[0] + (size_t)ox * actPitch[1] + (size_t)oy * actPitch[2] +
                          (size_t)oz * actPitch[3] + coord[4] * inCperG + fd;
          AccumulatorType F = static_cast<ElemType*>(weights)[index1];
          AccumulatorType I = static_cast<ElemType*>(activations)[index2];
          sum += (F - filterOffset) * (I - inOffset);
        }
      }
    }
  }

  // ElemTy & result = quantize(float(sum + B))
  //
  sum += B;

  float tmpLow, discarded, ignored;
  static_assert(sizeof(sum) == 8 or sizeof(sum) == 4);
  if constexpr (sizeof(sum) == 8) {
    float tmpHigh;
    __asm__ __volatile__("fbcx.ps %[tmpLow], %[low]\n"
                         "fbcx.ps %[tmpHigh], %[high]\n"
                         : [ tmpLow ] "=&f"(tmpLow), [ tmpHigh ] "=f"(tmpHigh)
                         : [ low ] "r"(sum), [ high ] "r"(sum >> 32));
    convert<Int64ITy, FloatTy>(tmpLow, tmpHigh, tmp, discarded, ignored, ignored, ignored, ignored);
  } else {
    __asm__ __volatile__("fbcx.ps %[tmpLow], %[low]\n" : [ tmpLow ] "=&f"(tmpLow) : [ low ] "r"(sum));
    convert<Int32ITy, FloatTy>(tmpLow, ignored, tmp, discarded, ignored, ignored, ignored, ignored);
  }

  doQuantize<dstElK>(tmp, tmp, outQuantScaleRec, outQuantOffset);
  int64_t first;
  __asm__ __volatile__("fmvs.x.ps %[first], %[tmp], 0\n" : [ first ] "=r"(first) : [ tmp ] "f"(tmp));
  result = static_cast<ElemType>(first);
}

template <ElemKind elK, ElemKind biasElK, size_t N, size_t PN>
INLINE_ATTR void convolution3DQuantizedInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                                            const std::array<uint32_t, N>& kernels,
                                            const std::array<uint32_t, N>& strides,
                                            const std::array<uint32_t, PN>& pads, unsigned int group, uint64_t flags,
                                            const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  void* output = outT->getRawDataPointer();
  void* activations = in1T->getRawDataPointer();
  void* weights = in2T->getRawDataPointer();
  void* bias = in3T->getRawDataPointer();

  const float inScale = in1T->getScale();
  const float filterScale = in2T->getScale();
  const float biasScale = in3T->getScale();
  const float outScale = outT->getScale();

  const int32_t inOffset = in1T->getOffset();
  const int32_t filterOffset = in2T->getOffset();
  const int32_t biasOffset = in3T->getOffset();
  const int32_t outOffset = outT->getOffset();

  const dim_t* dstIndex = outT->dims().data();
  const dim_t* actIndex = in1T->dims().data();
  const dim_t* dstPitch = outT->strides().data();
  const dim_t* actPitch = in1T->strides().data();
  const dim_t* weightPitch = in2T->strides().data();

  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, output);
  if (maxRead == 0)
    return;

  assert(actIndex[4] % group == 0 && "Input channels must be divisible by group.");
  assert(dstIndex[4] % group == 0 && "Output channels must be divisible by group.");
  auto inCperG = actIndex[4] / group;
  auto outCperG = dstIndex[4] / group;

  size_t eDstPitch[6] = {dstPitch[0], dstPitch[1], dstPitch[2], dstPitch[3], outCperG, 1};
  size_t eDstIndex[6] = {dstIndex[0], dstIndex[1], dstIndex[2], dstIndex[3], group, outCperG};

  dim_array_t coord = {0, 0, 0, 0, 0, 0};
  dim_t k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 6, eDstPitch, eDstIndex, k);

  size_t offsetOut = 0;
  for (size_t i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  auto posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, z, d;

  while (!done && (offsetOut < posMax)) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[2]);
    z = coord[3] * strides[2] - ssize_t(pads[4]);
    d = coord[4] * outCperG + coord[5];

    quantConvolution3DOp<elK, biasElK, N>(activations, weights, bias, output, offsetOut, coord, actPitch, weightPitch,
                                          actIndex, kernels, inCperG, x, y, z, d, inScale, filterScale, biasScale,
                                          outScale, inOffset, filterOffset, biasOffset, outOffset);

    done = getOffsets(6, coord, offsetOut, eDstIndex, eDstPitch);
  }

  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0)
    evict_va_multi(DO_EVICTS, (uintptr_t)output + typeSize * initialAddr, clperminion);
}

template <ElemKind elK, size_t N, size_t PN>
INLINE_ATTR void convolution3DNonQuantizedInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                                               const std::array<uint32_t, N>& kernels,
                                               const std::array<uint32_t, N>& strides,
                                               const std::array<uint32_t, PN>& pads, unsigned int group, uint64_t flags,
                                               const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  void* dstMatrix = outT->getRawDataPointer();
  void* activations = in1T->getRawDataPointer();
  void* weights = in2T->getRawDataPointer();
  void* bias = in3T->getRawDataPointer();

  Addresser<elK> tOutput(dstMatrix, outT->getScale(), outT->getOffset());
  const Addresser<elK> tAInput(activations, in1T->getScale(), in1T->getOffset());
  const Addresser<elK> tWInput(weights, in2T->getScale(), in2T->getOffset());
  const Addresser<elK> tBias(bias, in3T->getScale(), in3T->getOffset());

  const dim_t* dstIndex = outT->dims().data();
  const dim_t* actIndex = in1T->dims().data();
  const dim_t* dstPitch = outT->strides().data();
  const dim_t* actPitch = in1T->strides().data();
  const dim_t* weightPitch = in2T->strides().data();

  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dstMatrix);
  if (maxRead == 0)
    return;
  assert(actIndex[4] % group == 0 && "Input channels must be divisible by group.");
  assert(dstIndex[4] % group == 0 && "Output channels must be divisible by group.");
  auto inCperG = actIndex[4] / group;
  auto outCperG = dstIndex[4] / group;

  size_t eDstPitch[6] = {dstPitch[0], dstPitch[1], dstPitch[2], dstPitch[3], outCperG, 1};
  size_t eDstIndex[6] = {dstIndex[0], dstIndex[1], dstIndex[2], dstIndex[3], group, outCperG};

  dim_array_t coord = {0, 0, 0, 0, 0, 0};
  dim_t k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 6, eDstPitch, eDstIndex, k);

  size_t offsetOut = 0;
  for (size_t i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  auto posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, z, d;
  while (!done && (offsetOut < posMax)) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[2]);
    z = coord[3] * strides[2] - ssize_t(pads[4]);
    d = coord[4] * outCperG + coord[5];

    auto sum = tAInput[0];
    sum = 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {
      for (size_t fy = 0; fy < kernels[1]; fy++) {
        for (size_t fz = 0; fz < kernels[2]; fz++) {
          ssize_t ox = x + fx;
          ssize_t oy = y + fy;
          ssize_t oz = z + fz;

          // Ignore index access below zero (this is due to padding).
          if (ox < 0 || oy < 0 || oz < 0 || ox >= ssize_t(actIndex[1]) || oy >= ssize_t(actIndex[2]) ||
              oz >= ssize_t(actIndex[3])) {
            continue;
          }
          for (size_t fd = 0; fd < inCperG; fd++) {
            auto op1 =
              tWInput[d * weightPitch[0] + fx * weightPitch[1] + fy * weightPitch[2] + fz * weightPitch[3] + fd];
            auto op2 = tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] + (size_t)oy * actPitch[2] +
                               (size_t)oz * actPitch[3] + coord[4] * inCperG + fd];
            sum += op1 * op2;
          }
        }
      }
    }
    sum += tBias[d];
    tOutput[offsetOut] = sum;

    done = getOffsets(6, coord, offsetOut, eDstIndex, eDstPitch);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0)
    evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize * initialAddr, clperminion);
}

template <ElemKind elK, ElemKind biasElK, size_t N, size_t PN>
INLINE_ATTR void fwdLibConvolution3DInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                                         const std::array<uint32_t, N>& kernels, const std::array<uint32_t, N>& strides,
                                         const std::array<uint32_t, PN>& pads, unsigned int group, uint64_t flags,
                                         const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if constexpr (dnn_lib::isQuantizedElemKind(elK)) {
    convolution3DQuantizedInst<elK, biasElK, N, PN>(outT, in1T, in2T, in3T, kernels, strides, pads, group, flags,
                                                    minionOffset, assignedMinions);
  } else {
    convolution3DNonQuantizedInst<elK, N, PN>(outT, in1T, in2T, in3T, kernels, strides, pads, group, flags,
                                              minionOffset, assignedMinions);
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _CONVOLUTION_3D_INST_H_
