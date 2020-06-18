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

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "Float16.h"
#include "Writer.h" // From include/internal path
#include "Addresser.h" // From include/internal path
#include "Converter.h" // From include/internal path
#include "Operator.h" // From include/internal path
#include "utils.h" // From include/internal path
#include "LibTensor.h"


namespace dnn_lib {

namespace inlining {

#ifndef ACCUMULATOR_TYPE
#define ACCUMULATOR_TYPE

template<typename srcType>
struct accumulatorType {
  using type =
    typename std::conditional<std::is_same<srcType, int64_t>::value, int64_t,
      typename std::conditional<std::is_same<srcType, int32_t>::value, int32_t,
        float>::type >::type;
};

#endif

/**
 * @brief Performs the conolution operation between the activation, weights and bias.
 *
 * This convolution admits the division of the chanel into gropus and the use of stride
 * in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
 * The convolution is executed by the first minion only.
 * 
 * @tparam srcType Type of the elements of the tensors involved in the 
 *  convolution (except for the bias)
 * @param[out] dstMatrix Matrix in wich we save the result of the convolution.
 * @param[in] dstMatrixDims Vector of dimensions of the dstMatrix 
 *  (with batch and chanel).
 * @param[in] dstMatrixPitches Vector of pitches of the dstMatrix.
 * @param[in] weights Matrix with the weights for the convolution.
 * @param[in] weightDims Vector of dimensions of the weights. Unused.
 * @param[in] weightPitches Vector of pitches of the weights.
 * @param[in] bias Floats vector of biases (one for each chanel in a group).
 * @param[in] pkernels Vector of dimensions of the kernek that is applied.
 * @param[in] pstrides Vector with the strides for both dimensions.
 * @param[in] ppads Vector with the padding for both dimensions.
 * @param[in] group The number of groups in which we divide the chanel.
 * @param[in] scale The scale for the quantization.
 * @param[in] offset The offset for the quantization.
 */
template <ElemKind dstElK, ElemKind src1ElK, ElemKind src2ElK, size_t N>
inline void fwdLibConvolutionInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                  LibTensor* in3T,
                                  const std::array<uint32_t, N> &kernels,
                                  const std::array<uint32_t, N> &strides,
                                  const std::array<uint32_t, N> &pads,
                                  unsigned int group,
                                  uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  //  using dstType = typename elemKind2elemTy<dstElK>::type;
  //  using src1Type = typename elemKind2elemTy<src1ElK>::type;
  //  using src2Type = typename elemKind2elemTy<src2ElK>::type;
  
  // FIXME: going back to single thread until general case is solved with
  // multithread
  if (get_minion_id() != minionOffset) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT->dest in1T->activations in2T-> weight in3T->bias */
  void* dstMatrix = outT->getRawDataPointer<void>();
  void* activations = in1T->getRawDataPointer<void>();
  void* weights =  in2T->getRawDataPointer<void>();

  // Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  Addresser<dstElK> tOutput(dstMatrix, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<src1ElK> tAInput(activations, in1T->getScale(), in1T->getOffset());
  // const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  const Addresser<src2ElK> tWInput(weights, in2T->getScale(), in2T->getOffset());
  // float *tBias = (float *)bias;
  float *tBias = in3T->getRawDataPointer<float>();

  // unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const dim_t *actIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const dim_t *actPitch = in1T->strides().data();
  // unsigned int *weightPitch = (unsigned int *)weightPitches;
  const dim_t *weightPitch = in2T->strides().data();
  
  assert(actIndex[3] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 &&
         "Output channels must be divisible by group.");
  size_t inCperG = actIndex[3] / group;
  size_t outCperG = dstIndex[3] / group;

  // For each input in the batch:
  for (size_t n = 0; n < actIndex[0]; n++) {

    // For each group of input channels:
    for (size_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (size_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pads[0]);
        for (size_t ax = 0; ax < dstIndex[1]; x += strides[0], ax++) {
          ssize_t y = -ssize_t(pads[1]);
          for (size_t ay = 0; ay < dstIndex[2]; y += strides[1], ay++) {

            // For each element in the convolution-filter:
            auto sum = tAInput[0];
            sum = 0;
            for (size_t fx = 0; fx < kernels[0]; fx++) {
              for (size_t fy = 0; fy < kernels[1]; fy++) {
                ssize_t ox = x + fx;
                ssize_t oy = y + fy;

                // Ignore index access below zero (this is due to padding). The
                // elegance of this should be improved
                if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
                    oy >= ssize_t(actIndex[2])) {
                  continue;
                }

                for (size_t fd = 0; fd < inCperG; fd++) {
                  auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                                     fy * weightPitch[2] + fd];
                  auto op2 =
                      tAInput[n * actPitch[0] + (size_t)ox * actPitch[1] +
                              (size_t)oy * actPitch[2] + g * inCperG + fd];
                  sum += op1 * op2;
                }
              }
            }
            int64_t addr = n * dstPitch[0] + (size_t)ax * dstPitch[1] +
                           (size_t)ay * dstPitch[2] + d;
            sum += tBias[d];
            tOutput[addr] = sum;
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

#define CONVOLUTION_MAX_ELEMS 8

/**
 * @brief Computes one step in the convolution.
 *
 * This function accumulates in the result all the input channels
 * by its weight for a specific step in the kernel X and kernel Y.
 * 
 * @tparam srcType The type of the elements in the matrices.
 * @param[in] sum Pointer to tensor where the results must be accumulated
 * @param[in] tAInput Tensor where the activations are.
 * @param[in] tAInputPtr Pointer to activation tensor.
 * @param[in] tWInput Tensor where the weights are.
 * @param[in] tWInputPtr Pointer to weight tensor.
 * @param[in] inCperG Number of input channels to accumulate per output channel.
 * @param[in] actOffset Offset in the activation tensor pointing to data for this step.
 * @param[in] weightOffset Offset in the weight array where the weights for current step are.
 * @param[in] weightPitch Pitch of one row of the weight tensor.
 * @param[in] elems Number of elements in a row to compute.
 */
  template <ElemKind srcElK,
          typename std::enable_if<srcElK == Int64ITy, std::size_t>::type = 0>
inline void convolutionStep (int64_t *sum,
                             const Addresser<srcElK> &tAInput,
                             void * tAInputPtr,
                             const Addresser<srcElK> &tWInput,
                             void * tWInputPtr,
                             size_t inCperG,
                             size_t actOffset,
                             size_t weightOffset,
                             size_t weightPitch,
                             size_t elems) {
  // For all the input channels of the current "pixel"
  for (size_t dataF = 0; dataF < inCperG; dataF++) {
    // Gets input value
    auto act = tAInput[actOffset + dataF];
    for (size_t elem = 0; elem < elems; elem++) {
      // Gets weight value
      auto weight = tWInput[weightOffset + elem + dataF * weightPitch];
      // Adds to the result
      sum[elem] += act * weight;
    }
  }
}

/**
 * @brief Computes one step in the convolution.
 *
 * This function accumulates in the result all the input channels
 * by its weight for a specific step in the kernel X and kernel Y.
 * 
 * @tparam srcType The type of the elements in the matrices.
 * @param[in] sum Pointer to tensor where the results must be accumulated
 * @param[in] tAInput Tensor where the activations are.
 * @param[in] tAInputPtr Pointer to activation tensor.
 * @param[in] tWInput Tensor where the weights are.
 * @param[in] tWInputPtr Pointer to weight tensor.
 * @param[in] inCperG Number of input channels to accumulate per output channel.
 * @param[in] actOffset Offset in the activation tensor pointing to data for this step.
 * @param[in] weightOffset Offset in the weight array where the weights for current step are.
 * @param[in] weightPitch Pitch of one row of the weight tensor.
 * @param[in] elems Number of elements in a row to compute.
 */
  template <ElemKind srcElK,
          typename std::enable_if<srcElK == Int32ITy, std::size_t>::type = 0>
inline void convolutionStep (int32_t *sum,
                             const Addresser<srcElK> &tAInput,
                             void * tAInputPtr,
                             const Addresser<srcElK> &tWInput,
                             void * tWInputPtr,
                             size_t inCperG,
                             size_t actOffset,
                             size_t weightOffset,
                             size_t weightPitch,
                             size_t elems) {
  // For all the input channels of the current "pixel"
  for (size_t dataF = 0; dataF < inCperG; dataF++) {
    // Gets input value
    auto act = tAInput[actOffset + dataF];
    for (size_t elem = 0; elem < elems; elem++) {
      // Gets weight value
      auto weight = tWInput[weightOffset + elem + dataF * weightPitch];
      // Adds to the result
      sum[elem] += act * weight;
    }
  }
}

/**
 * @brief Computes one step in the convolution.
 *
 * This function accumulates in the result all the input channels
 * by its weight for a specific step in the kernel X and kernel Y.
 * 
 * @tparam srcType The type of the elements in the matrices.
 * @param[in] sum Pointer to tensor where the results must be accumulated
 * @param[in] tAInput Tensor where the activations are.
 * @param[in] tAInputPtr Pointer to activation tensor.
 * @param[in] tWInput Tensor where the weights are.
 * @param[in] tWInputPtr Pointer to weight tensor.
 * @param[in] inCperG Number of input channels to accumulate per output channel.
 * @param[in] actOffset Offset in the activation tensor pointing to data for this step.
 * @param[in] weightOffset Offset in the weight array where the weights for current step are.
 * @param[in] weightPitch Pitch of one row of the weight tensor.
 * @param[in] elems Number of elements in a row to compute.
 */
template <ElemKind srcElK>
inline void convolutionStep (float *sum,
                             const Addresser<srcElK> &tAInput,
                             void * tAInputPtr,
                             const Addresser<srcElK> &tWInput,
                             void * tWInputPtr,
                             size_t inCperG,
                             size_t actOffset,
                             size_t weightOffset,
                             size_t weightPitch,
                             size_t elems) {
  // Float version
  if (srcElK == FloatTy) {
    char * tAAddr = (char *) tAInputPtr;
    tAAddr += actOffset * 4;
    char * tWAddr = (char *) tWInputPtr;
    tWAddr += weightOffset * 4;

    // Computes the offsets for the gather of weight
    int32_t offsets[CONVOLUTION_MAX_ELEMS];
    for (size_t i = 0; i < CONVOLUTION_MAX_ELEMS; i++) {
      offsets[i] = i * weightPitch * 4;
    }

    // Active elements
    size_t mask = (1 << elems) - 1;
    __asm__ __volatile__(
        // Sets 1 lane enabled, moves scalar to float
        "mov.m.x	   mt0, %[mask], 0\n"
        "flw.ps      f2, 0(%[sum])\n"      // Loads initial value
        "flw.ps      f3, 0(%[offsets])\n"  // Loads offsets for gathers
        // Main loop
        "1:\n"
        "fbc.ps      f0, 0(%[tAAddr])\n"   // Loads data
        "fgw.ps      f1, f3(%[tWAddr])\n"
        "fmadd.ps    f2, f1, f0, f2\n"     // Accum
        // End of loop
        "addi   %[inCperG], %[inCperG], -1\n"
        "addi   %[tAAddr], %[tAAddr], 4\n"    // Increment pointers
        "addi   %[tWAddr], %[tWAddr], 4\n"
        "bne    %[inCperG], x0, 1b\n"
        // Converts back to scalar
        "fsw.ps f2, 0(%[sum])\n"
      : [tAAddr] "+&r" (tAAddr),
        [tWAddr] "+&r" (tWAddr),
        [inCperG] "+&r" (inCperG),
        [sum] "+&r" (sum)
      : [mask] "r" (mask),
        [offsets] "r" (offsets)
      : "memory", "f0", "f1", "f2"
    );
  }
  // Float16 version
  else if (srcElK == Float16Ty) {
    char * tAAddr = (char *) tAInputPtr;
    tAAddr += actOffset * 2;
    char * tWAddr = (char *) tWInputPtr;
    tWAddr += weightOffset * 2;

    // Computes the offsets for the gather of weight
    int32_t offsets[CONVOLUTION_MAX_ELEMS];
    for (size_t i = 0; i < CONVOLUTION_MAX_ELEMS; i++) {
      offsets[i] = i * weightPitch * 2;
    }

    // Active elements
    size_t mask = (1 << elems) - 1;
    __asm__ __volatile__(
        // Sets 1 lane enabled, moves scalar to float
        "mov.m.x	   mt0, %[mask], 0\n"
        "flw.ps      f2, 0(%[sum])\n"      // Loads initial value
        "flw.ps      f3, 0(%[offsets])\n"  // Loads offsets for gathers
        // Main loop
        "1:\n"
        "fbc.ps      f0, 0(%[tAAddr])\n"   // Loads data
        "fgh.ps      f1, f3(%[tWAddr])\n"
        "fcvt.ps.f16 f0, f0\n"             // Converts to FP32
        "fcvt.ps.f16 f1, f1\n"
        "fmadd.ps    f2, f1, f0, f2\n"     // Accum
        // End of loop
        "addi   %[inCperG], %[inCperG], -1\n"
        "addi   %[tAAddr], %[tAAddr], 2\n"    // Increment pointers
        "addi   %[tWAddr], %[tWAddr], 2\n"
        "bne    %[inCperG], x0, 1b\n"
        // Converts back to scalar
        "fsw.ps f2, 0(%[sum])\n"
      : [tAAddr] "+&r" (tAAddr),
        [tWAddr] "+&r" (tWAddr),
        [inCperG] "+&r" (inCperG),
        [sum] "+&r" (sum)
      : [mask] "r" (mask),
        [offsets] "r" (offsets)
      : "memory", "f0", "f1", "f2"
    );
  }
  // Others
  else {
    // For all the input channels of the current "pixel"
    for (size_t dataF = 0; dataF < inCperG; dataF++) {
      // Gets input value
      auto act = tAInput[actOffset + dataF];
      for (size_t elem = 0; elem < elems; elem++) {
        // Gets weight value
        auto weight = tWInput[weightOffset + dataF + elem  * weightPitch];
        // Adds to the result
        sum[elem] += act * weight;
      }
    }
  }
}

/**
 * @brief Performs the convolution operation between the activation, weights and bias.
 *
 * This convolution admits the division of the chanel into gropus and the use of stride
 * in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
 * This is the threaded version for the convolution.
 * 
 * @tparam srcType Type of the elements of the tensors involved in the 
 *  convolution (except for the bias)
 * @param[out] outT Tensor where we save the result of the convolution.
 * @param[in] in1T Tensor with the activations of the convolution.
 * @param[in] in2T Tensor with the weights of the convolution.
 * @param[in] in2T Tensor with the biases of the convolution.
 * @param[in] pkernels Vector of dimensions of the kernel that is applied.
 * @param[in] pstrides Vector with the strides for both dimensions.
 * @param[in] ppads Vector with the padding for both dimensions.
 * @param[in] group The number of groups in which we divide the chanel.
 * @param[in] flags Controls the active shires and the type of evict that 
 *  should be done at the end of the function.
 */
  template <ElemKind dstElK, ElemKind src1ElK, ElemKind src2ElK, size_t N>
inline void fwdLibConvolutionInstThreaded(LibTensor* outT, LibTensor* in1T,
                                          LibTensor* in2T, LibTensor* in3T,
                                          const std::array<uint32_t, N> &kernels,
                                          const std::array<uint32_t, N> &strides,
                                          const std::array<uint32_t, N> &pads,
                                          unsigned int group,
                                          uint64_t flags,
             const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using dstType = typename elemKind2elemTy<dstElK>::type;
  using src1Type = typename elemKind2elemTy<src1ElK>::type;
  //  using src2Type = typename elemKind2elemTy<src2ElK>::type;
  
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT->dest in1T->activations in2T-> weight in3T->bias */
  void* dstMatrix   = outT->getRawDataPointer<void>();
  void* activations = in1T->getRawDataPointer<void>();
  void* weights     = in2T->getRawDataPointer<void>();

  Addresser<dstElK>       tOutput(dstMatrix, outT->getScale(), outT->getOffset());
  const Addresser<src1ElK> tAInput(activations, in1T->getScale(), in1T->getOffset());
  const Addresser<src2ElK> tWInput(weights, in2T->getScale(), in2T->getOffset());
  float *tBias = in3T->getRawDataPointer<float>();

  const dim_t *dstIndex = outT->dims().data();
  const dim_t *actIndex = in1T->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *actPitch = in1T->strides().data();
  const dim_t *weightPitch = in2T->strides().data();
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  assert(actIndex[3] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 &&
         "Output channels must be divisible by group.");
  unsigned int inCperG = actIndex[3] / group;
  unsigned int outCperG = dstIndex[3] / group;

  size_t eDstPitch[5] = {dstPitch[0], dstPitch[1], dstPitch[2], outCperG, 1};

  size_t eDstIndex[5] = {dstIndex[0], dstIndex[1], dstIndex[2], group, outCperG};

  unsigned int coord[5], k;
  getNonPaddingCoordinates(coord, initialAddr, 5, eDstPitch, eDstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  while ((offsetOut < posMax) && !done) {
    // Computes position of the pixel being computed
    size_t dataB = coord[0];                                 // Batch
    size_t dataX = coord[1] * strides[0] - ssize_t(pads[0]); // X
    size_t dataY = coord[2] * strides[1] - ssize_t(pads[1]); // Y
    size_t dataC = coord[3];                                 // Out Channel

    // Computes position of the weight channel
    size_t weightC = coord[3] * outCperG + coord[4];

    // Precomputes data to move out of loop
    size_t dataBOffset   = dataB * actPitch[0];
    size_t dataCOffset   = dataC * inCperG;
    size_t dataBCOffset  = dataBOffset + dataCOffset;
    size_t weightCOffset = weightC * weightPitch[0];

    // Computes how many more elements can we compute
    size_t elems = CONVOLUTION_MAX_ELEMS;
    // Can't go beyond the elements left
    size_t elemsLeft = posMax - offsetOut;
    if(elems > elemsLeft) { elems = elemsLeft; }
    // Can't go beyond current row
    size_t featsLeft = eDstIndex[4] - coord[4];
    if(elems > featsLeft) { elems = featsLeft; }

    // Starts the accumulation with the bias (per Channel)
    typename accumulatorType<src1Type>::type sum[CONVOLUTION_MAX_ELEMS];
    for (size_t i = 0; i < elems; i++) {
      sum[i] = tBias[weightC + i];
    }

    // For all the rows of the kernel
    for (size_t kernelXStep = 0; kernelXStep < kernels[0]; kernelXStep++) {
      ssize_t dataKernelX = dataX + kernelXStep; // Position of the pixel sampled for the convolution

      // Ignore index accesses outside the "image"
      if (
            (dataKernelX < 0)                     // Left
         || (dataKernelX >= ssize_t(actIndex[1])) // Right
      ) {
        continue;
      }

      // Precompute offsets
      size_t dataKernelXOffset        = dataKernelX * actPitch[1];
      size_t dataBCKernelXOffset      = dataBCOffset + dataKernelXOffset;
      size_t weightKernelXStepOffset  = kernelXStep * weightPitch[1];
      size_t weightKernelXStepCOffset = weightCOffset + weightKernelXStepOffset;

      // For all the cols of the kernel
      for (size_t kernelYStep = 0; kernelYStep < kernels[1]; kernelYStep++) {
        ssize_t dataKernelY = dataY + kernelYStep;

        // Precomputes offsets
        ssize_t dataKernelYOffset       = dataKernelY * actPitch[2];
        ssize_t dataBCKernelOffset      = dataBCKernelXOffset + dataKernelYOffset;
        ssize_t weightKernelYStepOffset = kernelYStep * weightPitch[2];
        ssize_t weightKernelStepCOffset = weightKernelXStepCOffset + weightKernelYStepOffset;

        // Ignore index accesses outside the "image"
        if (
              (dataKernelY < 0)                     // Up
           || (dataKernelY >= ssize_t(actIndex[2])) // Down
        ) {
          continue;
        }

        // Calls the function that will accumulate all the channels for specific kernel step
        convolutionStep <src1ElK> (sum, tAInput, activations, tWInput, weights, inCperG, dataBCKernelOffset, weightKernelStepCOffset, weightPitch[0], elems);
      }
    }
    // Moves to next result
    for (size_t i = 0; i < elems; i++) {
      tOutput[offsetOut] = sum[i];
      done = getOffsets(5, coord, offsetOut, eDstIndex, eDstPitch);
    }
  }

  // Check if evicts required
  if (DO_EVICTS) {
    unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
    if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
  }
}

/**
 * @brief Computes one element in the convolution.
 *
 * This consists on the vectorized implementation for the products of 
 * convolutionInst, which works computing the product of the elements 
 * in the filter with the activations in groups of up to 8 elements and 
 * sums them together at the end.
 * 
 * @tparam src1Type The type of the elements in the src1 matrix, which in this funcion is
 *  is imposed to be float.
 * @tparam src2Type The type of the elements in the src2 matrix.
 * @tparam dstType The type of the elements in the dst matrix.
 * @param[in] activations Matrix of activations for the convolution.
 * @param[in] weights Matrix of weights for the convolution.
 * @param[in] coord The vector of coordinates to the initial position in the 
 *  activations. coord[0] corresponds to the batch and coord[3] corresponds 
 *  to the group where we are.
 * @param[in] actPitch Vector of pitches of the activations matrix.
 * @param[in] weightPitch Vector of pitches of the weights matrix.
 * @param[in] actIndex Vector of the size of each dimensions of the activations.
 * @param[in] kernels Dimensions of the filters or kernels.
 * @param[in] inCperG Elements in a group.
 * @param[out] sum The result of applying the filter in the given position.
 * @param[in] mask The int32_t that determines which lanes should be active when 
 *  we can't take 8 elements at the same time.
 * @param[in] x, y, d Coordinates where our minions should start reading.
 */
template <ElemKind src1ElK, ElemKind src2ElK, ElemKind dstElK, size_t N,
          typename std::enable_if<src1ElK == FloatTy, std::size_t>::type = 0>
inline void convolutionOp (void *activations, void *weights, unsigned int *coord,
                           const dim_t *actPitch, const dim_t *weightPitch,
                           const dim_t *actIndex, const std::array<uint32_t, N> &kernels,
                           unsigned int inCperG, float &sum, int32_t mask, ssize_t x,
                           ssize_t y, ssize_t d, const float *scale, const int32_t *offset) {
  int64_t dist;
  ssize_t fx, fy, ox, oy;
  fx = fy = 0;
  unsigned int *actAddr = (unsigned int *) activations;
  unsigned int *weightAddr = (unsigned int *) weights;
  actAddr += coord[0] * actPitch[0] + x * actPitch[1] + y * actPitch[2] +
            coord[3] * inCperG;
  weightAddr += d * weightPitch[0];
  __asm__ __volatile__(
    "fxor.pi  f0, f0, f0\n"                         // f0 to zeros
    "mov.m.x  m0, zero, 0xff\n"                     // m0 to ones
    "mov.m.x  m1, %[mask], 0\n"                     // m1 the auxiliar mask
    "1:\n"                                          // for (size_t fx = 0; fx < kernels[0]; fx++) {
    "beq      %[fy], zero, 2f\n"
    "mul      %[fy], %[kernels1], %[actPitch2]\n"
    "sub      %[actAddr], %[actAddr], %[fy]\n"
    "mul      %[fy], %[kernels1], %[weightPitch2]\n"
    "sub      %[weightAddr], %[weightAddr], %[fy]\n"
    "addi     %[fy], zero, 0\n"
    "2:\n"                                            // for (size_t fy = 0; fy < kernels[1]; fy++) {
    "addi     %[dist], %[inCperG], 0\n"                // dist = inCperG

    "add      %[oy], %[y], %[fy]\n"                     // oy = y + fy
    "add      %[ox], %[x], %[fx]\n"                     // ox = x + fx

    "blt      %[ox], zero, 5f\n"                        // if (ox < 0) continue
    "blt      %[oy], zero, 5f\n"                        // if (oy < 0) continue
    "ble      %[actIndex1], %[ox], 5f\n"                // if (actIndex[1] <= ox) continue
    "ble      %[actIndex2], %[oy], 5f\n"                // if (actIndex[2] <= oy) continue

    "addi     t0, zero, 8\n"                            // t0 = 8
    "ble      %[dist], t0, 4f\n"                        // if dist <= 8 go to 4

    "mov.m.x  m0, zero, 0xff\n"
    "3:\n"                                              // while (8 < dist) {
    "flw.ps   f1, 0x0(%[actAddr])\n"                      // actAddr -> f1
    "flw.ps   f2, 0x0(%[weightAddr])\n"                   // weightaddr -> f2
    "fmadd.ps f0, f1, f2, f0\n"                           // f0 = (f1 * f2) + f0
    "addi     %[actAddr], %[actAddr], 32\n"               // actAddr += 32
    "addi     %[weightAddr], %[weightAddr], 32\n"         // weightAddr += 32
    "addi     %[dist], %[dist], -8\n"                     // dist -= 8
    "blt      t0, %[dist], 3b\n"                        // }

    "4:\n"
    "maskand  m0, m0, m1\n"                             // put mask on
    "flw.ps   f1, 0x0(%[actAddr])\n"                    // actAddr -> f1
    "flw.ps   f2, 0x0(%[weightAddr])\n"                 // weightaddr -> f2
    "fmadd.ps f0, f1, f2, f0\n"                         // f0 = (f1 * f2) + f0
    "sub      %[dist], %[inCperG], %[dist]\n"           // dist = inCperG - dist
    "slli     %[dist], %[dist], 2\n"                    // dist = dist * 4
    "sub      %[actAddr], %[actAddr], %[dist]\n"        // actAddr = actAddr - dist
    "sub      %[weightAddr], %[weightAddr], %[dist]\n"  // actAddr = actAddr - dist

    "5:\n"
    "addi     %[fy], %[fy], 1\n"                        // fy++
    "add     %[actAddr], %[actPitch2], %[actAddr]\n"   // actAddr = actAddr + actPitch[2]
    "add     %[weightAddr], %[weightPitch2], %[weightAddr]\n"
    "blt      %[fy], %[kernels1], 2b\n"               // Closing fy for }

    "addi     %[fx], %[fx], 1\n"                      // fx++

    "add     %[actAddr], %[actPitch1], %[actAddr]\n" // actAddr = actAddr + actPitch[1]
    "add     %[weightAddr], %[weightPitch1], %[weightAddr]\n"
    "blt      %[fx], %[kernels0], 1b\n"             // Closing fx for{}

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

    : [ weightAddr ] "+&r" (weightAddr),
      [ actAddr ] "+&r" (actAddr),
      [ dist ] "=&r" (dist),
      [ sum ] "+&r" (sum),
      [ ox ] "=&r" (ox),
      [ oy ] "=&r" (oy),
      [ fy ] "+&r" (fy),
      [ fx ] "+&r" (fx)
    : [ weightPitch1 ] "r" (weightPitch[1] * 4),
      [ weightPitch2 ] "r" (weightPitch[2] * 4),
      [ actIndex1 ] "r" (actIndex[1]),
      [ actIndex2 ] "r" (actIndex[2]),
      [ actPitch1 ] "r" (actPitch[1] * 4),
      [ actPitch2 ] "r" (actPitch[2] * 4),
      [ kernels0 ] "r" (kernels[0]),
      [ kernels1 ] "r" (kernels[1]),
      [ inCperG ] "r" (inCperG),
      [ mask ] "r" (mask),
      [ x ] "r" (x),
      [ y ] "r" (y)
    : "memory", "f0", "f1", "f2", "f31", "t0", "t1");
  return;
}

/**
 * @brief Computes one element in the convolution.
 *
 * @overload
 */
template <ElemKind src1ElK, ElemKind src2ElK, ElemKind dstElK, size_t N,
          typename std::enable_if<src1ElK == Float16Ty, std::size_t>::type = 0>
inline void convolutionOp (void *activations, void *weights, unsigned int *coord,
                           const dim_t *actPitch, const dim_t *weightPitch,
                           const dim_t *actIndex, const std::array<uint32_t, N> &kernels,
                           unsigned int inCperG, float16 &sum, int32_t mask, ssize_t x,
                           ssize_t y, ssize_t d, const float *scale, const int32_t *offset) {
  int dist;
  ssize_t fx, fy, ox, oy;
  fx = fy = 0;
  uint16_t *actAddr = (uint16_t *) activations;
  uint16_t *weightAddr = (uint16_t *) weights;
  actAddr += coord[0] * actPitch[0] + x * actPitch[1] + y * actPitch[2] +
            coord[3] * inCperG;
  weightAddr += d * weightPitch[0];
  unsigned int gatherValues[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };
  __asm__ __volatile__(
    "flw.ps f16, 0x0(%[gatherValues])\n"
    "fxor.pi  f0, f0, f0\n"                         // f0 to zeros
    "mov.m.x  m0, zero, 0xff\n"                     // m0 to ones
    "mov.m.x  m1, %[mask], 0\n"                     // m1 the auxiliar mask
    "1:\n"                                          // for (size_t fx = 0; fx < kernels[0]; fx++) {
    "beq      %[fy], zero, 2f\n"
    "mul      %[fy], %[kernels1], %[actPitch2]\n"
    "sub      %[actAddr], %[actAddr], %[fy]\n"
    "mul      %[fy], %[kernels1], %[weightPitch2]\n"
    "sub      %[weightAddr], %[weightAddr], %[fy]\n"
    "addi     %[fy], zero, 0\n"
    "2:\n"                                            // for (size_t fy = 0; fy < kernels[1]; fy++) {
    "addi     %[dist], %[inCperG], 0\n"                // dist = inCperG

    "add      %[oy], %[y], %[fy]\n"                     // oy = y + fy
    "add      %[ox], %[x], %[fx]\n"                     // ox = x + fx

    "blt      %[ox], zero, 5f\n"                        // if (ox < 0) continue
    "blt      %[oy], zero, 5f\n"                        // if (oy < 0) continue
    "ble      %[actIndex1], %[ox], 5f\n"                // if (actIndex[1] <= ox) continue
    "ble      %[actIndex2], %[oy], 5f\n"                // if (actIndex[2] <= oy) continue

    "addi     t0, zero, 8\n"                            // t0 = 8
    "ble      %[dist], t0, 4f\n"                        // if dist <= 8 go to 4

    "mov.m.x  m0, zero, 0xff\n"
    "3:\n"                                              // while (8 < dist) {
    "fgh.ps   f1, f16(%[actAddr])\n"                      // actAddr -> f1
    "fcvt.ps.f16 f1, f1\n"
    "fgh.ps   f2, f16(%[weightAddr])\n"                   // weightaddr -> f2
    "fcvt.ps.f16 f2, f2\n"
    "fmadd.ps f0, f1, f2, f0\n"                           // f0 = (f1 * f2) + f0
    "addi     %[actAddr], %[actAddr], 16\n"               // actAddr += 16
    "addi     %[weightAddr], %[weightAddr], 16\n"         // weightAddr += 16
    "addi     %[dist], %[dist], -8\n"                     // dist -= 8
    "blt      t0, %[dist], 3b\n"                        // }

    "4:\n"
    "maskand  m0, m0, m1\n"                             // put mask on
    "fgh.ps   f1, f16(%[actAddr])\n"                    // actAddr -> f1
    "fcvt.ps.f16 f1, f1\n"
    "fgh.ps   f2, f16(%[weightAddr])\n"                 // weightaddr -> f2
    "fcvt.ps.f16 f2, f2\n"
    "fmadd.ps f0, f1, f2, f0\n"                         // f0 = (f1 * f2) + f0
    "sub      %[dist], %[inCperG], %[dist]\n"           // dist = inCperG - dist
    "slli     %[dist], %[dist], 1\n"                    // dist = dist * 2
    "sub      %[actAddr], %[actAddr], %[dist]\n"        // actAddr = actAddr - dist
    "sub      %[weightAddr], %[weightAddr], %[dist]\n"  // actAddr = actAddr - dist

    "5:\n"
    "addi     %[fy], %[fy], 1\n"                        // fy++
    "add      %[actAddr], %[actPitch2], %[actAddr]\n"   // actAddr = actAddr + actPitch[2]
    "add      %[weightAddr], %[weightPitch2], %[weightAddr]\n"
    "blt      %[fy], %[kernels1], 2b\n"               // Closing fy for{}

    "addi     %[fx], %[fx], 1\n"                      // fx++

    "add      %[actAddr], %[actPitch1], %[actAddr]\n" // actAddr = actAddr + actPitch[1]
    "add      %[weightAddr], %[weightPitch1], %[weightAddr]\n"
    "blt      %[fx], %[kernels0], 1b\n"             // Closing fx for{}

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

    : [ weightAddr ] "+&r" (weightAddr),
      [ actAddr ] "+&r" (actAddr),
      [ dist ] "+&r" (dist),
      [ sum ] "+&r" (sum),
      [ ox ] "+&r" (ox),
      [ oy ] "+&r" (oy),
      [ fy ] "+&r" (fy),
      [ fx ] "+&r" (fx)
    : [ weightPitch1 ] "r" (weightPitch[1] * 2),
      [ weightPitch2 ] "r" (weightPitch[2] * 2),
      [ gatherValues ] "r" (gatherValues),
      [ actPitch1 ] "r" (actPitch[1] * 2),
      [ actPitch2 ] "r" (actPitch[2] * 2),
      [ actIndex1 ] "r" (actIndex[1]),
      [ actIndex2 ] "r" (actIndex[2]),
      [ kernels0 ] "r" (kernels[0]),
      [ kernels1 ] "r" (kernels[1]),
      [ inCperG ] "r" (inCperG),
      [ mask ] "r" (mask),
      [ x ] "r" (x),
      [ y ] "r" (y)
    : "memory", "f0", "f1", "f2", "f31", "t0", "t1");
  return;
}

/**
 * @brief Computes one element in the convolution.
 *
 * This consists on the non-vectorized implementation for the products of 
 * convolutionInst, which is the same as in the threaded version, but works
 * for all the non supported types in the vectorized version of this same 
 * function.
 * 
 * @tparam src1Type The type of the elements in the src1 matrix.
 * @tparam src2Type The type of the elements in the src2 matrix.
 * @tparam dstType The type of the elements in the dst matrix.
 * @param[in] activations Matrix of activations for the convolution.
 * @param[in] weights Matrix of weights for the convolution.
 * @param[in] coord The vector of coordinates to the initial position in the 
 *  activations. coord[0] corresponds to the batch and coord[3] corresponds 
 *  to the group where we are.
 * @param[in] actPitch Vector of pitches of the activations matrix.
 * @param[in] weightPitch Vector of pitches of the weights matrix.
 * @param[in] actIndex Vector of the size of each dimensions of the activations.
 * @param[in] kernels Dimensions of the filters or kernels.
 * @param[in] inCperG Elements in a group.
 * @param[out] sum The result of applying the filter in the given position.
 * @param[in] mask It has no relevance in this function.
 * @param[in] x, y, d Coordinates where our minions should start reading.
 */
template <ElemKind src1ElK, ElemKind src2ElK, ElemKind dstElK, size_t N,
            typename std::enable_if<src1ElK != FloatTy, std::size_t>::type = 0>
inline void convolutionOp (void *activations, void *weights, unsigned int *coord,
                           const dim_t *actPitch, const dim_t *weightPitch,
                           const dim_t *actIndex, const std::array<uint32_t, N> &kernels,
                           unsigned int inCperG, float &sum, int32_t mask, ssize_t x,
                           ssize_t y, ssize_t d, const float *scale, const int32_t *offset) {
  const Addresser<src1ElK> tAInput(activations, scale[0], offset[0]);
  const Addresser<src2ElK> tWInput(weights, scale[1], offset[1]);
  for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }
  return; //TODO return error.
}

/**
 * @brief Computes one element in the convolution.
 *
 * @overload
 */
template <ElemKind src1ElK, ElemKind src2ElK, ElemKind dstElK, size_t N,
          typename std::enable_if<src1ElK != Float16Ty, std::size_t>::type = 0>
inline void convolutionOp (void *activations, void *weights, unsigned int *coord,
                           const dim_t *actPitch, const dim_t *weightPitch,
                           const dim_t *actIndex, const std::array<uint32_t, N> &kernels,
                           unsigned int inCperG, float16 &sum, int32_t mask, ssize_t x,
                           ssize_t y, ssize_t d, const float *scale, const int32_t *offset) {
  const Addresser<src1ElK> tAInput(activations, scale[0], offset[0]);
  const Addresser<src2ElK> tWInput(weights, scale[1], offset[1]);
  for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }
  return; //TODO return error.
}


  template <ElemKind src1ElK, ElemKind src2ElK, ElemKind dstElK, size_t N>
inline void convolutionOp (void *activations, void *weights, unsigned int *coord,
                           const dim_t *actPitch, const dim_t *weightPitch,
                           const dim_t *actIndex, const std::array<uint32_t, N> &kernels,
                           unsigned int inCperG, int32_t &sum, int32_t mask, ssize_t x,
                           ssize_t y, ssize_t d, const float *scale, const int32_t *offset) {
  const Addresser<src1ElK> tAInput(activations, scale[0], offset[0]);
  const Addresser<src2ElK> tWInput(weights, scale[1], offset[1]);
  for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
    for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }
  return; //TODO return error.
}


/**
 * @brief Performs the convolution operation between the activation, weights and bias.
 *
 * This convolution admits the division of the chanel into gropus and the use of stride
 * in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
 * This is the threaded and vectorized version for the convolution.
 * 
 * @tparam src1Type Type of the elements of the src1 tensor involved in the 
 *  convolution (except for the bias)
 * @tparam src2Type Type of the elements of the src2 tensor involved in the 
 *  convolution (except for the bias)
 * @tparam dstType Type of the elements of the dst tensor involved in the 
 *  convolution (except for the bias)
 * @param[out] dstMatrix Matrix in wich we save the result of the convolution.
 * @param[in] dstMatrixDims Vector of dimensions of the dstMatrix 
 *  (with batch and chanel).
 * @param[in] dstMatrixPitches Vector of pitches of the dstMatrix.
 * @param[in] weights Matrix with the weights for the convolution.
 * @param[in] weightDims Vector of dimensions of the weights. Unused.
 * @param[in] weightPitches Vector of pitches of the weights.
 * @param[in] bias Floats vector of biases (one for each chanel in a group).
 * @param[in] pkernels Vector of dimensions of the kernek that is applied.
 * @param[in] pstrides Vector with the strides for both dimensions.
 * @param[in] ppads Vector with the padding for both dimensions.
 * @param[in] group The number of groups in which we divide the chanel.
 * @param[in] scale The scale for the quantization.
 * @param[in] offset The offset for the quantization.
 * @param[in] flags Controls the active shires and the type of evict that 
 *  should be done at the end of the function.
 */
template <ElemKind dstElK, ElemKind src1ElK, ElemKind src2ElK, size_t N>
inline void fwdLibConvolutionInstVectorized(LibTensor* outT, LibTensor* in1T,
                                            LibTensor* in2T, LibTensor* in3T,
                                            const std::array<uint32_t, N> &kernels,
                                            const std::array<uint32_t, N> &strides,
                                            const std::array<uint32_t, N> &pads,
                                            unsigned int group,
                                            uint64_t flags,
                                            const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  //  using dstType = typename elemKind2elemTy<dstElK>::type;
  using src1Type = typename elemKind2elemTy<src1ElK>::type;
  //  using src2Type = typename elemKind2elemTy<src2ElK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT->dest in1T->activations in2T-> weight in3T->bias */

  void *dstMatrix = outT->getRawDataPointer<void>();
  void *activations = in1T->getRawDataPointer<void>();
  void *weights = in2T->getRawDataPointer<void>();

  Addresser<dstElK> tOutput(dstMatrix, outT->getScale(), outT->getOffset());  
  // float *tBias = (float *)bias;
  float *tBias = in3T->getRawDataPointer<float>();
  
  // unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const dim_t *actIndex = in1T->dims().data();

  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const dim_t *actPitch = in1T->strides().data();
  // unsigned int *weightPitch = (unsigned int *)weightPitches;
  const dim_t *weightPitch = in2T->strides().data();

  float scale[] = { in1T->getScale(), in2T->getScale(), in3T->getScale(), outT->getScale()};
  int32_t offset[] = { in1T->getOffset(), in2T->getOffset(), in3T->getOffset(), outT->getOffset()};
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<src1Type>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  assert(actIndex[3] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 &&
         "Output channels must be divisible by group.");
  unsigned int inCperG = actIndex[3] / group;
  unsigned int outCperG = dstIndex[3] / group;

  // unsigned int eDstPitch[5] = {dstPitch[0], dstPitch[1], dstPitch[2], outCperG,
  //                              1};

  // unsigned int eDstIndex[5] = {dstIndex[0], dstIndex[1], dstIndex[2], group,
  //                              outCperG};
  dim_t eDstPitch[5] = {dstPitch[0], dstPitch[1], dstPitch[2], outCperG,
                               1};

  dim_t eDstIndex[5] = {dstIndex[0], dstIndex[1], dstIndex[2], group,
                               outCperG};

  unsigned int coord[5], k;
  getNonPaddingCoordinates(coord, initialAddr, 5, eDstPitch, eDstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, d;
  int32_t mask = (1 << (((inCperG - 1) & 0x7)  + 1)) - 1;

  while ((offsetOut < posMax) && !done) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);
    d = coord[3] * outCperG + coord[4];

    auto sum = tBias[d];
    convolutionOp <src1ElK, src2ElK, dstElK> (activations, weights, coord, actPitch, weightPitch,
                                              actIndex, kernels, inCperG, sum, mask, x, y, d,
                                              scale, offset);
    tOutput[offsetOut] = sum;

    done = getOffsets(5, coord, offsetOut, eDstIndex, eDstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _CONVOLUTION_INST_H_
