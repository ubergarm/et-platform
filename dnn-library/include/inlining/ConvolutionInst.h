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

#ifndef _CONVOLUTION_INST_H_
#define _CONVOLUTION_INST_H_

#include "Addresser.h"
#include "Float16.h"
#include "LibCommon.h"
#include "LibTensor.h"
#include "LibTypes.h"
#include "LibUtils.h"
#include "LoadStore.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

/// \brief Computes one element in the convolution.
///
/// This consists on the vectorized implementation for the products of
/// convolutionInst, which works computing the product of the elements
/// in the filter with the activations in groups of up to 8 elements and
/// sums them together at the end.
///
/// \tparam dstElK Element kind of source and destination tensors
/// \tparam biasElK Element kind of bias
/// \tparam N Size of kernel_size array (compiler-provided through automatic template deduction)
/// \param[in] activations Matrix of activations for the convolution.
/// \param[in] weights Matrix of weights for the convolution.
/// \param[in] coord The vector of coordinates to the initial position in the
///  activations. coord[0] corresponds to the batch and coord[3] corresponds
///  to the group where we are.
/// \param[in] actPitch Vector of pitches of the activations matrix.
/// \param[in] weightPitch Vector of pitches of the weights matrix.
/// \param[in] actIndex Vector of the size of each dimensions of the activations.
/// \param[in] kernels Dimensions of the filters or kernels.
/// \param[in] inCperG Elements in a group.
/// \param[out] sum The result of applying the filter in the given position.
/// \param[in] mask Which lanes should be active
/// \param[in] x, y, d Coordinates where our minions should start reading.

template <ElemKind dstElK, size_t N, typename std::enable_if<dstElK == FloatTy, std::size_t>::type = 0>
INLINE_ATTR void convolutionOp(void* activations, void* weights, const dim_array_t& coord, const dim_t* actPitch,
                               const dim_t* weightPitch, const dim_t* actIndex, const std::array<uint32_t, N>& kernels,
                               dim_t inCperG, float& sum, int32_t mask, ssize_t x, ssize_t y, ssize_t d,
                               const float* scale, const int32_t* offset, const std::array<uint32_t, N> dilation) {

  (void)offset;
  (void)scale;

  int64_t dist;
  ssize_t fx, fy, ox, oy;
  fx = fy = 0;
  unsigned int* actAddr = (unsigned int*)activations;
  unsigned int* weightAddr = (unsigned int*)weights;
  actAddr += coord[0] * actPitch[0] + x * actPitch[1] + y * actPitch[2] + coord[3] * inCperG;
  weightAddr += d * weightPitch[0];
  __asm__ __volatile__("mov.m.x  m0, zero, 0xff\n" // m0 to ones
                       "mov.m.x  m1, %[mask], 0\n" // m1 the auxiliar mask
                       "fxor.pi  f0, f0, f0\n"     // f0 to zeros
                       "1:\n"                      // for (size_t fx = 0; fx < kernels[0]; fx++) {
                       "beq      %[fy], zero, 2f\n"
                       "mul      %[fy], %[kernels1], %[actPitch2]\n"
                       "sub      %[actAddr], %[actAddr], %[fy]\n"
                       "mul      %[fy], %[kernels1], %[weightPitch2]\n"
                       "sub      %[weightAddr], %[weightAddr], %[fy]\n"
                       "addi     %[fy], zero, 0\n"
                       "2:\n"                              // for (size_t fy = 0; fy < kernels[1]; fy++) {
                       "addi     %[dist], %[inCperG], 0\n" // dist = inCperG

                       "mulw     %[oy], %[fy], %[dilation1]\n" // oy = y + fy *dilation
                       "add      %[oy], %[oy], %[y]\n"
                       "blt      %[oy], zero, 5f\n" // if (oy < 0) continue

                       "mulw     %[ox], %[fx], %[dilation0]\n" // ox = x + fx *dilation
                       "add      %[ox], %[ox], %[x]\n"
                       "blt      %[ox], zero, 5f\n" // if (ox < 0) continue

                       "ble      %[actIndex1], %[ox], 5f\n" // if (actIndex[1] <= ox) continue
                       "ble      %[actIndex2], %[oy], 5f\n" // if (actIndex[2] <= oy) continue

                       "mov.m.x  m0, zero, 0xff\n"

                       "addi     t0, zero, 8\n"     // t0 = 8
                       "ble      %[dist], t0, 4f\n" // if dist <= 8 go to 4

                       "3:\n"                                        // while (8 < dist) {
                       "flw.ps   f1, 0x0(%[actAddr])\n"              // actAddr -> f1
                       "flw.ps   f2, 0x0(%[weightAddr])\n"           // weightaddr -> f2
                       "fmadd.ps f0, f1, f2, f0\n"                   // f0 = (f1 * f2) + f0
                       "addi     %[actAddr], %[actAddr], 32\n"       // actAddr += 32
                       "addi     %[weightAddr], %[weightAddr], 32\n" // weightAddr += 32
                       "addi     %[dist], %[dist], -8\n"             // dist -= 8
                       "blt      t0, %[dist], 3b\n"                  // }

                       "4:\n"
                       "maskand  m0, m0, m1\n"                            // put mask on
                       "flw.ps   f1, 0x0(%[actAddr])\n"                   // actAddr -> f1
                       "flw.ps   f2, 0x0(%[weightAddr])\n"                // weightaddr -> f2
                       "fmadd.ps f0, f1, f2, f0\n"                        // f0 = (f1 * f2) + f0
                       "sub      %[dist], %[inCperG], %[dist]\n"          // dist = inCperG - dist
                       "slli     %[dist], %[dist], 2\n"                   // dist = dist * 4
                       "sub      %[actAddr], %[actAddr], %[dist]\n"       // actAddr = actAddr - dist
                       "sub      %[weightAddr], %[weightAddr], %[dist]\n" // actAddr = actAddr - dist

                       "5:\n"
                       "addi     %[fy], %[fy], 1\n"                     // fy++
                       "add     %[actAddr], %[actPitch2], %[actAddr]\n" // actAddr = actAddr + actPitch[2]
                       "add     %[weightAddr], %[weightPitch2], %[weightAddr]\n"
                       "blt      %[fy], %[kernels1], 2b\n" // Closing fy for }

                       "addi     %[fx], %[fx], 1\n" // fx++

                       "add     %[actAddr], %[actPitch1], %[actAddr]\n" // actAddr = actAddr + actPitch[1]
                       "add     %[weightAddr], %[weightPitch1], %[weightAddr]\n"
                       "blt      %[fx], %[kernels0], 1b\n" // Closing fx for{}

                       "mov.m.x   m0, zero, 0xff\n"
                       "fswizz.ps f1, f0, 0xe\n"
                       "fadd.ps   f0, f0, f1\n"
                       "fswizz.ps f1, f0, 0x1\n"
                       "fadd.ps   f0, f0, f1\n"
                       "fmvs.x.ps t0, f0, 0x4\n"
                       "fmv.w.x   f31, t0\n"
                       "fadd.s    f31, f31, f0\n"

                       "fmv.w.x   f0, %[sum]\n"
                       "fadd.s    f31, f31, f0\n"
                       "fmv.x.w   %[sum], f31\n"

                       : [ weightAddr ] "+&r"(weightAddr), [ actAddr ] "+&r"(actAddr), [ dist ] "=&r"(dist),
                         [ sum ] "+&r"(sum), [ ox ] "=&r"(ox), [ oy ] "=&r"(oy), [ fy ] "+&r"(fy), [ fx ] "+&r"(fx)
                       : [ weightPitch1 ] "r"(weightPitch[1] * 4), [ weightPitch2 ] "r"(weightPitch[2] * 4),
                         [ actIndex1 ] "r"(actIndex[1]), [ actIndex2 ] "r"(actIndex[2]),
                         [ actPitch1 ] "r"(actPitch[1] * dilation[0] * 4),
                         [ actPitch2 ] "r"(actPitch[2] * dilation[1] * 4), [ kernels0 ] "r"(kernels[0]),
                         [ kernels1 ] "r"(kernels[1]), [ inCperG ] "r"(inCperG), [ mask ] "r"(mask), [ x ] "r"(x),
                         [ y ] "r"(y), [ dilation0 ] "r"(dilation[0]), [ dilation1 ] "r"(dilation[1])
                       : "memory", "f0", "f1", "f2", "f31", "t0", "t1");
}

/// \brief Computes one element in the convolution.

template <ElemKind dstElK, size_t N, typename std::enable_if<dstElK == Float16Ty, std::size_t>::type = 0>
INLINE_ATTR void convolutionOp(void* activations, void* weights, const dim_array_t& coord, const dim_t* actPitch,
                               const dim_t* weightPitch, const dim_t* actIndex, const std::array<uint32_t, N>& kernels,
                               dim_t inCperG, float16& sum, int32_t mask, ssize_t x, ssize_t y, ssize_t d,
                               const float* scale, const int32_t* offset, const std::array<uint32_t, N>& dilation) {
  (void)offset;
  (void)scale;

  int dist;
  ssize_t fx, fy, ox, oy;
  fx = fy = 0;
  uint16_t* actAddr = (uint16_t*)activations;
  uint16_t* weightAddr = (uint16_t*)weights;
  actAddr += coord[0] * actPitch[0] + x * actPitch[1] + y * actPitch[2] + coord[3] * inCperG;
  weightAddr += d * weightPitch[0];
  unsigned int gatherValues[8] = {0, 2, 4, 6, 8, 10, 12, 14};
  __asm__ __volatile__("mov.m.x  m0, zero, 0xff\n" // m0 to ones
                       "mov.m.x  m1, %[mask], 0\n" // m1 the auxiliar mask
                       "flw.ps f16, 0x0(%[gatherValues])\n"
                       "fxor.pi  f0, f0, f0\n" // f0 to zeros
                       "1:\n"                  // for (size_t fx = 0; fx < kernels[0]; fx++) {
                       "beq      %[fy], zero, 2f\n"
                       "mul      %[fy], %[kernels1], %[actPitch2]\n"
                       "sub      %[actAddr], %[actAddr], %[fy]\n"
                       "mul      %[fy], %[kernels1], %[weightPitch2]\n"
                       "sub      %[weightAddr], %[weightAddr], %[fy]\n"
                       "addi     %[fy], zero, 0\n"
                       "2:\n"                              // for (size_t fy = 0; fy < kernels[1]; fy++) {
                       "addi     %[dist], %[inCperG], 0\n" // dist = inCperG

                       "mulw     %[oy], %[fy], %[dilation1]\n" // oy = y + fy *dilation
                       "add      %[oy], %[oy], %[y]\n"
                       "blt      %[oy], zero, 5f\n" // if (oy < 0) continue

                       "mulw     %[ox], %[fx], %[dilation0]\n" // ox = x + fx *dilation
                       "add      %[ox], %[ox], %[x]\n"
                       "blt      %[ox], zero, 5f\n" // if (ox < 0) continue

                       "ble      %[actIndex1], %[ox], 5f\n" // if (actIndex[1] <= ox) continue
                       "ble      %[actIndex2], %[oy], 5f\n" // if (actIndex[2] <= oy) continue

                       "mov.m.x  m0, zero, 0xff\n"

                       "addi     t0, zero, 8\n"     // t0 = 8
                       "ble      %[dist], t0, 4f\n" // if dist <= 8 go to 4

                       "3:\n"                           // while (8 < dist) {
                       "fgh.ps   f1, f16(%[actAddr])\n" // actAddr -> f1
                       "fcvt.ps.f16 f1, f1\n"
                       "fgh.ps   f2, f16(%[weightAddr])\n" // weightaddr -> f2
                       "fcvt.ps.f16 f2, f2\n"
                       "fmadd.ps f0, f1, f2, f0\n"                   // f0 = (f1 * f2) + f0
                       "addi     %[actAddr], %[actAddr], 16\n"       // actAddr += 16
                       "addi     %[weightAddr], %[weightAddr], 16\n" // weightAddr += 16
                       "addi     %[dist], %[dist], -8\n"             // dist -= 8
                       "blt      t0, %[dist], 3b\n"                  // }

                       "4:\n"
                       "maskand  m0, m0, m1\n"          // put mask on
                       "fgh.ps   f1, f16(%[actAddr])\n" // actAddr -> f1
                       "fcvt.ps.f16 f1, f1\n"
                       "fgh.ps   f2, f16(%[weightAddr])\n" // weightaddr -> f2
                       "fcvt.ps.f16 f2, f2\n"
                       "fmadd.ps f0, f1, f2, f0\n"                        // f0 = (f1 * f2) + f0
                       "sub      %[dist], %[inCperG], %[dist]\n"          // dist = inCperG - dist
                       "slli     %[dist], %[dist], 1\n"                   // dist = dist * 2
                       "sub      %[actAddr], %[actAddr], %[dist]\n"       // actAddr = actAddr - dist
                       "sub      %[weightAddr], %[weightAddr], %[dist]\n" // actAddr = actAddr - dist

                       "5:\n"
                       "addi     %[fy], %[fy], 1\n"                      // fy++
                       "add      %[actAddr], %[actPitch2], %[actAddr]\n" // actAddr = actAddr + actPitch[2]
                       "add      %[weightAddr], %[weightPitch2], %[weightAddr]\n"
                       "blt      %[fy], %[kernels1], 2b\n" // Closing fy for{}

                       "addi     %[fx], %[fx], 1\n" // fx++

                       "add      %[actAddr], %[actPitch1], %[actAddr]\n" // actAddr = actAddr + actPitch[1]
                       "add      %[weightAddr], %[weightPitch1], %[weightAddr]\n"
                       "blt      %[fx], %[kernels0], 1b\n" // Closing fx for{}

                       "mov.m.x   m0, zero, 0xff\n"
                       "fswizz.ps f1, f0, 0xe\n"
                       "fadd.ps   f0, f0, f1\n"
                       "fswizz.ps f1, f0, 0x1\n"
                       "fadd.ps   f0, f0, f1\n"
                       "fmvs.x.ps t0, f0, 0x4\n"
                       "fmv.w.x   f31, t0\n"
                       "fadd.s    f31, f31, f0\n"
                       "fmv.w.x   f0, %[sum]\n"
                       "fadd.s    f31, f31, f0\n"
                       "fmv.x.w   %[sum], f31\n"

                       : [ weightAddr ] "+&r"(weightAddr), [ actAddr ] "+&r"(actAddr), [ dist ] "+&r"(dist),
                         [ sum ] "+&r"(sum), [ ox ] "+&r"(ox), [ oy ] "+&r"(oy), [ fy ] "+&r"(fy), [ fx ] "+&r"(fx)
                       : [ weightPitch1 ] "r"(weightPitch[1] * 2), [ weightPitch2 ] "r"(weightPitch[2] * 2),
                         [ gatherValues ] "r"(gatherValues), [ actPitch1 ] "r"(actPitch[1] * dilation[0] * 2),
                         [ actIndex1 ] "r"(actIndex[1]), [ actPitch2 ] "r"(actPitch[2] * dilation[1] * 2),
                         [ actIndex2 ] "r"(actIndex[2]), [ kernels0 ] "r"(kernels[0]), [ kernels1 ] "r"(kernels[1]),
                         [ inCperG ] "r"(inCperG), [ mask ] "r"(mask), [ x ] "r"(x), [ y ] "r"(y),
                         [ dilation0 ] "r"(dilation[0]), [ dilation1 ] "r"(dilation[1])
                       : "memory", "f0", "f1", "f2", "f31", "t0", "t1");
}

/// \brief Computes one element in the convolution.
///
/// This consists on the non-vectorized implementation for the products of
/// convolutionInst, which is the same as in the threaded version, but works
/// for all the non supported types in the vectorized version of this same
/// function.
///
/// \tparam dstElK Element kind of source and destination tensors
/// \tparam biasElK Element kind of bias
/// \tparam N Size of kernel_size array (compiler-provided through automatic template deduction)
/// \param[in] activations Matrix of activations for the convolution.
/// \param[in] weights Matrix of weights for the convolution.
/// \param[in] coord The vector of coordinates to the initial position in the
///  activations. coord[0] corresponds to the batch and coord[3] corresponds
///  to the group where we are.
/// \param[in] actPitch Vector of pitches of the activations matrix.
/// \param[in] weightPitch Vector of pitches of the weights matrix.
/// \param[in] actIndex Vector of the size of each dimensions of the activations.
/// \param[in] kernels Dimensions of the filters or kernels.
/// \param[in] inCperG Elements in a group.
/// \param[out] sum The result of applying the filter in the given position.
/// \param[in] mask It has no relevance in this function.
/// \param[in] x, y, d Coordinates where our minions should start reading.

template <ElemKind dstElK, size_t N, typename std::enable_if<dstElK != FloatTy, std::size_t>::type = 0>
INLINE_ATTR void convolutionOp(void* activations, void* weights, const dim_array_t& coord, const dim_t* actPitch,
                               const dim_t* weightPitch, const dim_t* actIndex, const std::array<uint32_t, N>& kernels,
                               dim_t inCperG, float& sum, [[maybe_unused]] int32_t mask, ssize_t x, ssize_t y,
                               ssize_t d, const float* scale, const int32_t* offset,
                               const std::array<uint32_t, N>& dilation) {

  const Addresser<dstElK> tAInput(activations, scale[0], offset[0]);
  const Addresser<dstElK> tWInput(weights, scale[1], offset[1]);
  for (size_t fx = 0; fx < kernels[0]; fx++) {   // for all x coordinates in kernel
    for (size_t fy = 0; fy < kernels[1]; fy++) { // for all y coordinates in kernel
      ssize_t ox = x + fx * dilation[0];
      ssize_t oy = y + fy * dilation[1];

      // Ignore index access below zero (this is due to padding).
      if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) || oy >= ssize_t(actIndex[2])) {
        continue;
      }
      for (size_t fd = 0; fd < inCperG; fd++) { // for all depth coordinates
        auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] + fy * weightPitch[2] + fd];
        auto op2 = tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] + (size_t)oy * actPitch[2] +
                           coord[3] * inCperG + fd];
        sum += op1 * op2;
      }
    }
  }
}

/// \brief Computes one output element on the convolution.
///
/// This code is not vectorized.
///
/// \tparam dstElK Element kind of source and destination tensors
/// \tparam biasElK Element kind of bias
/// \tparam N Size of kernel_size array (compiler-provided through automatic template deduction)
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
/// \param[in] mask It has no relevance in this function.
/// \param[in] x, y, d Coordinates where our minions should start reading.
/// \param[in] scale Array of tensor scales
/// \param[in] offset Array of tensor offsets
/// \param[in] dilation Array of dilations

template <ElemKind dstElK, ElemKind biasElK, size_t N>
INLINE_ATTR void quantConvolutionOp(void* activations, void* weights, void* bias, void* output, size_t offsetOut,
                                    const dim_array_t& coord, const dim_t* actPitch, const dim_t* weightPitch,
                                    const dim_t* actIndex, const std::array<uint32_t, N>& kernels, dim_t inCperG,
                                    [[maybe_unused]] int32_t mask, ssize_t x, ssize_t y, ssize_t d, const float* scale,
                                    const int32_t* offset, const std::array<uint32_t, N>& dilation) {

  // This code assumes the vector mask is 1 because it is not vectorized
  assert(mask == 1);
  __asm__ __volatile__("mov.m.x m0, zero, 0x1\n");

  using ElemType = typename AccumulatingQuantizedOpTypes<dstElK, biasElK>::elemType;
  using AccumulatorType = typename AccumulatingQuantizedOpTypes<dstElK, biasElK>::accumulatorType;
  using BiasType = typename AccumulatingQuantizedOpTypes<dstElK, biasElK>::biasType;

  const float& inScale = scale[0];
  const float& filterScale = scale[1];
  const float& biasScale = scale[2];
  const float& outScale = scale[3];

  const int32_t& inOffset = offset[0];
  const int32_t& filterOffset = offset[1];
  const int32_t& biasOffset = offset[2];
  const int32_t& outOffset = offset[3];

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
  for (size_t fx = 0; fx < kernels[0]; fx++) {   // for all x coordinates in kernel
    for (size_t fy = 0; fy < kernels[1]; fy++) { // for all y coordinates in kernel
      ssize_t ox = x + fx * dilation[0];
      ssize_t oy = y + fy * dilation[1];

      // Ignore index access below zero (this is due to padding).
      if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) || oy >= ssize_t(actIndex[2])) {
        continue;
      }
      for (size_t fd = 0; fd < inCperG; fd++) { // for all depth coordinates
        size_t index1 = d * weightPitch[0] + fx * weightPitch[1] + fy * weightPitch[2] + fd;
        size_t index2 =
          coord[0] * actPitch[0] + (size_t)ox * actPitch[1] + (size_t)oy * actPitch[2] + coord[3] * inCperG + fd;
        AccumulatorType F = static_cast<ElemType*>(weights)[index1];
        AccumulatorType I = static_cast<ElemType*>(activations)[index2];
        sum += (F - filterOffset) * (I - inOffset);
      }
    }
  }

  // ElemTy & result = quantize(float(sum + B))
  //
  ElemType& result = static_cast<ElemType*>(output)[offsetOut];

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

/// \brief Performs a convolution
///
/// This implementation is threaded.
///
/// \tparam dstElK Element kind of source and destination tensors
/// \tparam biasElK Element kind of bias
/// \tparam N Size of kernel_size array (compiler-provided through automatic template deduction)
/// \tparam PN Size of padding array (compiler-provided through automatic template deduction)
/// \tparam FN Size of fusedActivationArgs (compiler-provided through automatic template deduction)
/// \param[out] dstMatrix Matrix in wich we save the result of the convolution.
/// \param[in] dstMatrixDims Vector of dimensions of the dstMatrix
///  (with batch and chanel).
/// \param[in] dstMatrixPitches Vector of pitches of the dstMatrix.
/// \param[in] weights Matrix with the weights for the convolution.
/// \param[in] weightDims Vector of dimensions of the weights. Unused.
/// \param[in] weightPitches Vector of pitches of the weights.
/// \param[in] bias Floats vector of biases (one for each chanel in a group).
/// \param[in] pkernels Vector of dimensions of the kernek that is applied.
/// \param[in] pstrides Vector with the strides for both dimensions.
/// \param[in] ppads Vector with the padding for both dimensions.
/// \param[in] group The number of groups in which we divide the chanel.
/// \param[in] scale The scale for the quantization.
/// \param[in] offset The offset for the quantization.
/// \param[in] flags Controls the active shires and the type of evict that
///  should be done at the end of the function.

template <ElemKind dstElK, ElemKind biasElK, size_t N, size_t PN, size_t FN>
INLINE_ATTR void
convolutionInstQuantized(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                         const std::array<uint32_t, N>& kernels, const std::array<uint32_t, N>& strides,
                         const std::array<uint32_t, PN>& pads, dim_t group, const std::array<uint32_t, N>& dilation,
                         [[maybe_unused]] const size_t fusedActivation,
                         [[maybe_unused]] const std::array<float, FN>& fusedActivationArgs, uint64_t flags,
                         const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  void* output = outT->getRawDataPointer();
  void* activations = in1T->getRawDataPointer();
  void* weights = in2T->getRawDataPointer();
  void* bias = in3T->getRawDataPointer();

  const dim_t* dstIndex = outT->dims().data();
  const dim_t* actIndex = in1T->dims().data();

  const dim_t* dstPitch = outT->strides().data();
  const dim_t* actPitch = in1T->strides().data();
  const dim_t* weightPitch = in2T->strides().data();

  float scale[] = {in1T->getScale(), in2T->getScale(), in3T->getScale(), outT->getScale()};
  int32_t offset[] = {in1T->getOffset(), in2T->getOffset(), in3T->getOffset(), outT->getOffset()};

  auto numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  using ElemType = typename elemKind2elemTy<dstElK>::type;
  size_t typeSize = getsize<ElemType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, output);
  if (maxRead == 0)
    return;

  assert(actIndex[3] % group == 0 && "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 && "Output channels must be divisible by group.");
  auto inCperG = actIndex[3] / group;
  auto outCperG = dstIndex[3] / group;

  dim_t eDstPitch[5] = {dstPitch[0], dstPitch[1], dstPitch[2], outCperG, 1};

  dim_t eDstIndex[5] = {dstIndex[0], dstIndex[1], dstIndex[2], group, outCperG};

  dim_array_t coord = {0};
  dim_t k;
  getNonPaddingCoordinates(coord, initialAddr, 5, eDstPitch, eDstIndex, k);

  size_t offsetOut = 0;
  for (size_t i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  auto posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, d;
  int32_t mask = (1 << (((inCperG - 1) & 0x7) + 1)) - 1;

  while ((offsetOut < posMax) && !done) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);
    d = coord[3] * outCperG + coord[4];

    quantConvolutionOp<dstElK, biasElK, N>(activations, weights, bias, output, offsetOut, coord, actPitch, weightPitch,
                                           actIndex, kernels, inCperG, mask, x, y, d, scale, offset, dilation);

    done = getOffsets(5, coord, offsetOut, eDstIndex, eDstPitch);
  }

  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0)
    evict_va_multi(DO_EVICTS, (uintptr_t)output + typeSize * initialAddr, clperminion);
}

/// \brief Performs the convolution operation between the activation, weights and bias.
///
/// This convolution admits the division of the chanel into gropus and the use of stride
/// in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
/// This is the threaded and vectorized version for the convolution.
///
/// \tparam dstElK Element kind of source and destination tensors
/// \tparam biasElK Element kind of bias
/// \tparam N Size of kernel_size array (compiler-provided through automatic template deduction)
/// \tparam PN Size of padding array (compiler-provided through automatic template deduction)
/// \tparam FN Size of fusedActivationArgs (compiler-provided through automatic template deduction)
/// \param[out] dstMatrix Matrix in wich we save the result of the convolution.
/// \param[in] dstMatrixDims Vector of dimensions of the dstMatrix
///  (with batch and chanel).
/// \param[in] dstMatrixPitches Vector of pitches of the dstMatrix.
/// \param[in] weights Matrix with the weights for the convolution.
/// \param[in] weightDims Vector of dimensions of the weights. Unused.
/// \param[in] weightPitches Vector of pitches of the weights.
/// \param[in] bias Floats vector of biases (one for each chanel in a group).
/// \param[in] pkernels Vector of dimensions of the kernek that is applied.
/// \param[in] pstrides Vector with the strides for both dimensions.
/// \param[in] ppads Vector with the padding for both dimensions.
/// \param[in] group The number of groups in which we divide the chanel.
/// \param[in] scale The scale for the quantization.
/// \param[in] offset The offset for the quantization.
/// \param[in] flags Controls the active shires and the type of evict that
///  should be done at the end of the function.

template <ElemKind dstElK, ElemKind biasElK, size_t N, size_t PN, size_t FN>
INLINE_ATTR void
convolutionInstNonQuantized(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                            const std::array<uint32_t, N>& kernels, const std::array<uint32_t, N>& strides,
                            const std::array<uint32_t, PN>& pads, dim_t group, const std::array<uint32_t, N>& dilation,
                            [[maybe_unused]] const size_t fusedActivation,
                            [[maybe_unused]] const std::array<float, FN>& fusedActivationArgs, uint64_t flags,
                            const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  using dstType = typename elemKind2elemTy<dstElK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  void* dstMatrix = outT->getRawDataPointer();
  void* activations = in1T->getRawDataPointer();
  void* weights = in2T->getRawDataPointer();
  void* bias = in3T->getRawDataPointer();

  Addresser<dstElK> tOutput(dstMatrix, outT->getScale(), outT->getOffset());
  const Addresser<biasElK> tBias(bias, in3T->getScale(), in3T->getOffset());
  const dim_t* dstIndex = outT->dims().data();
  const dim_t* actIndex = in1T->dims().data();

  const dim_t* dstPitch = outT->strides().data();
  const dim_t* actPitch = in1T->strides().data();
  const dim_t* weightPitch = in2T->strides().data();

  float scale[] = {in1T->getScale(), in2T->getScale(), in3T->getScale(), outT->getScale()};
  int32_t offset[] = {in1T->getOffset(), in2T->getOffset(), in3T->getOffset(), outT->getOffset()};

  auto numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dstMatrix);
  if (maxRead == 0)
    return;

  assert(actIndex[3] % group == 0 && "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 && "Output channels must be divisible by group.");
  dim_t inCperG = actIndex[3] / group;
  dim_t outCperG = dstIndex[3] / group;

  dim_t eDstPitch[5] = {dstPitch[0], dstPitch[1], dstPitch[2], outCperG, 1};

  dim_t eDstIndex[5] = {dstIndex[0], dstIndex[1], dstIndex[2], group, outCperG};

  dim_array_t coord = {0};
  dim_t k;
  getNonPaddingCoordinates(coord, initialAddr, 5, eDstPitch, eDstIndex, k);

  size_t offsetOut = 0;
  for (size_t i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  auto posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, d;
  int32_t mask = (1 << (((inCperG - 1) & 0x7) + 1)) - 1;

  while ((offsetOut < posMax) && !done) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);
    d = coord[3] * outCperG + coord[4];

    auto sum = tBias[d];
    convolutionOp<dstElK>(activations, weights, coord, actPitch, weightPitch, actIndex, kernels, inCperG, sum, mask, x,
                          y, d, scale, offset, dilation);
    tOutput[offsetOut] = sum;

    done = getOffsets(5, coord, offsetOut, eDstIndex, eDstPitch);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0)
    evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize * initialAddr, clperminion);
}

/// \brief Performs the convolution operation between the activation, weights and bias.
///
/// This convolution admits the division of the chanel into gropus and the use of stride
/// in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
/// This is the threaded and vectorized version for the convolution.
///
/// \tparam dstElK Element kind of source and destination tensors
/// \tparam biasElK Element kind of bias
/// \tparam N Size of kernel_size array (compiler-provided through automatic template deduction)
/// \tparam PN Size of padding array (compiler-provided through automatic template deduction)
/// \tparam FN Size of fusedActivationArgs (compiler-provided through automatic template deduction)
/// \param[out] dstMatrix Matrix in wich we save the result of the convolution.
/// \param[in] dstMatrixDims Vector of dimensions of the dstMatrix
///  (with batch and chanel).
/// \param[in] dstMatrixPitches Vector of pitches of the dstMatrix.
/// \param[in] weights Matrix with the weights for the convolution.
/// \param[in] weightDims Vector of dimensions of the weights. Unused.
/// \param[in] weightPitches Vector of pitches of the weights.
/// \param[in] bias Floats vector of biases (one for each chanel in a group).
/// \param[in] pkernels Vector of dimensions of the kernek that is applied.
/// \param[in] pstrides Vector with the strides for both dimensions.
/// \param[in] ppads Vector with the padding for both dimensions.
/// \param[in] group The number of groups in which we divide the chanel.
/// \param[in] scale The scale for the quantization.
/// \param[in] offset The offset for the quantization.
/// \param[in] flags Controls the active shires and the type of evict that
///  should be done at the end of the function.

template <ElemKind dstElK, ElemKind biasElK, size_t N, size_t PN, size_t FN>
INLINE_ATTR void fwdLibConvolutionInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                                       const std::array<uint32_t, N>& kernels, const std::array<uint32_t, N>& strides,
                                       const std::array<uint32_t, PN>& pads, dim_t group,
                                       const std::array<uint32_t, N>& dilation, const size_t fusedActivation,
                                       const std::array<float, FN>& fusedActivationArgs, uint64_t flags,
                                       const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  // SW-1110: enable the quantized code path for Int16QTy when the ticket is resolved
  if constexpr (dnn_lib::isQuantizedElemKind(dstElK) and dstElK != Int16QTy) {
    convolutionInstQuantized<dstElK, biasElK, N, PN, FN>(outT, in1T, in2T, in3T, kernels, strides, pads, group,
                                                         dilation, fusedActivation, fusedActivationArgs, flags,
                                                         minionOffset, assignedMinions);
  } else {
    convolutionInstNonQuantized<dstElK, biasElK, N, PN, FN>(outT, in1T, in2T, in3T, kernels, strides, pads, group,
                                                            dilation, fusedActivation, fusedActivationArgs, flags,
                                                            minionOffset, assignedMinions);
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _CONVOLUTION_INST_H_
