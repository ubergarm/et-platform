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

#ifndef _FULLY_CONNECTED_INST_H_
#define _FULLY_CONNECTED_INST_H_

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

#define FULLYCONNECTED_MAX_ELEMS 8

/**
 * @brief Computes one step (result) of a matmul.
 *
 * This function computes one result of a bigger matmul.
 * 
 * @tparam srcType The type of the elements in the matrices.
 * @param[in] sum Pointer to tensor where the results must be accumulated
 * @param[in] tAInput Tensor where the activations are.
 * @param[in] tAInputPtr Pointer to activation tensor.
 * @param[in] tWInput Tensor where the weights are.
 * @param[in] tWInputPtr Pointer to weight tensor.
 * @param[in] aCols Number of input channels to accumulate per output channel.
 * @param[in] actOffset Offset in the activation tensor pointing to data for this step.
 * @param[in] weightOffset Offset in the weight tensor where the weights for current step are.
 * @param[in] weightPitch Pitch of one row of the weight tensor.
 * @param[in] elems Number of elements in a row to compute.
 */
template <ElemKind srcElK,
          typename std::enable_if<srcElK == Int64ITy, std::size_t>::type = 0>
inline void matmulStep (int64_t *sum,
                        const Addresser<srcElK> &tAInput,
                        void * tAInputPtr,
                        const Addresser<srcElK> &tWInput,
                        void * tWInputPtr,
                        size_t aCols,
                        size_t actOffset,
                        size_t weightOffset,
                        size_t weightPitch,
                        size_t elems) {
  // For all the accumulations
  for (size_t aCol = 0; aCol < aCols; aCol++) {
    // Gets input value
    auto act = tAInput[actOffset + aCol];
    for (size_t elem = 0; elem < elems; elem++) {
      // Gets weight value
      auto weight = tWInput[weightOffset + elem + aCol * weightPitch];
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
 * @param[in] aCols Number of input channels to accumulate per output channel.
 * @param[in] actOffset Offset in the activation tensor pointing to data for this step.
 * @param[in] weightOffset Offset in the weight tensor where the weights for current step are.
 * @param[in] weightPitch Pitch of one row of the weight tensor.
 * @param[in] elems Number of elements in a row to compute.
 */
template <ElemKind srcElK,
          typename std::enable_if<srcElK == Int32ITy, std::size_t>::type = 0>
inline void matmulStep (int32_t *sum,
                        const Addresser<srcElK> &tAInput,
                        void * tAInputPtr,
                        const Addresser<srcElK> &tWInput,
                        void * tWInputPtr,
                        size_t aCols,
                        size_t actOffset,
                        size_t weightOffset,
                        size_t weightPitch,
                        size_t elems) {
  // For all the accumulations
  for (size_t aCol = 0; aCol < aCols; aCol++) {
    // Gets input value
    auto act = tAInput[actOffset + aCol];
    for (size_t elem = 0; elem < elems; elem++) {
      // Gets weight value
      auto weight = tWInput[weightOffset + elem + aCol * weightPitch];
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
 * @param[in] aCols Number of input channels to accumulate per output channel.
 * @param[in] actOffset Offset in the activation tensor pointing to data for this step.
 * @param[in] weightOffset Offset in the weight tensor where the weights for current step are.
 * @param[in] weightPitch Pitch of one row of the weight tensor.
 * @param[in] elems Number of elements in a row to compute.
 */
template <ElemKind srcElK>
inline void matmulStep (float *sum,
                        const Addresser<srcElK> &tAInput,
                        void * tAInputPtr,
                        const Addresser<srcElK> &tWInput,
                        void * tWInputPtr,
                        size_t aCols,
                        size_t actOffset,
                        size_t weightOffset,
                        size_t weightPitch,
                        size_t elems) {

  // Float version
  if (srcElK == FloatTy) {
    int size = 4;
    char * tAAddr = (char *) tAInputPtr;
    tAAddr += actOffset * size;
    char * tWAddr = (char *) tWInputPtr;
    tWAddr += weightOffset * size;
    weightPitch *= size;
    int offsets[FULLYCONNECTED_MAX_ELEMS];
    for (size_t i = 0; i < FULLYCONNECTED_MAX_ELEMS; i++) {
      offsets[i] = i * size;
    }
    size_t mask = (1 << elems) - 1;
    __asm__ __volatile__(
        // Sets 1 lane enabled, moves scalar to float
        "mov.m.x     mt0, %[mask], 0\n"
        "flw.ps      f2, 0(%[sum])\n"     // Loads initial value
        "flw.ps      f3, 0(%[offsets])\n" // Loads offsets for gathers
        // Main loop
        "1:\n"
        "fbc.ps      f0, 0(%[tAAddr])\n"   // Loads data
        "flw.ps      f1, 0(%[tWAddr])\n"
        "fmadd.ps    f2, f1, f0, f2\n"     // Accum
        // End of loop
        "addi   %[aCols], %[aCols], -1\n"
        "add    %[tAAddr], %[tAAddr], %[size]\n"              // Increment pointers
        "add    %[tWAddr], %[tWAddr], %[weightPitch]\n"
        "bne    %[aCols], x0, 1b\n"
        // Copies back to memory
        "fsw.ps f2, 0(%[sum])\n"
      : [tAAddr] "+&r" (tAAddr),
        [tWAddr] "+&r" (tWAddr),
        [aCols] "+&r" (aCols),
        [sum] "+&r" (sum)
      : [weightPitch] "r" (weightPitch),
        [mask] "r" (mask),
        [offsets] "r" (offsets),
        [size] "r" (size)
      : "memory", "f0", "f1", "f2", "f3"
    );
  }
  // Float16 version
  else if (srcElK == Float16Ty) {
    int size = 2;
    char * tAAddr = (char *) tAInputPtr;
    tAAddr += actOffset * size;
    char * tWAddr = (char *) tWInputPtr;
    tWAddr += weightOffset * size;
    weightPitch *= size;
    int offsets[FULLYCONNECTED_MAX_ELEMS];
    for (size_t i = 0; i < FULLYCONNECTED_MAX_ELEMS; i++) {
      offsets[i] = i * size;
    }
    size_t mask = (1 << elems) - 1;
    __asm__ __volatile__(
        // Sets 1 lane enabled, moves scalar to float
        "mov.m.x     mt0, %[mask], 0\n"
        "flw.ps      f2, 0(%[sum])\n"     // Loads initial value
        "flw.ps      f3, 0(%[offsets])\n" // Loads offsets for gathers
        // Main loop
        "1:\n"
        "fbc.ps      f0, 0(%[tAAddr])\n"   // Loads data
        "fgh.ps      f1, f3(%[tWAddr])\n"
        "fcvt.ps.f16 f0, f0\n"             // Converts to FP32
        "fcvt.ps.f16 f1, f1\n"
        "fmadd.ps    f2, f1, f0, f2\n"     // Accum
        // End of loop
        "addi   %[aCols], %[aCols], -1\n"
        "add    %[tAAddr], %[tAAddr], %[size]\n"              // Increment pointers
        "add    %[tWAddr], %[tWAddr], %[weightPitch]\n"
        "bne    %[aCols], x0, 1b\n"
        // Copies back to memory
        "fsw.ps f2, 0(%[sum])\n"
      : [tAAddr] "+&r" (tAAddr),
        [tWAddr] "+&r" (tWAddr),
        [aCols] "+&r" (aCols),
        [sum] "+&r" (sum)
      : [weightPitch] "r" (weightPitch),
        [mask] "r" (mask),
        [offsets] "r" (offsets),
        [size] "r" (size)
      : "memory", "f0", "f1", "f2", "f3"
    );
  }
  // Int8QTy version
  else if (srcElK == Int8QTy) {
    int size = 1;
    char * tAAddr = (char *) tAInputPtr;
    tAAddr += actOffset * size;
    char * tWAddr = (char *) tWInputPtr;
    tWAddr += weightOffset * size;
    weightPitch *= size;
    int gatherOffsetsA[FULLYCONNECTED_MAX_ELEMS];
    int gatherOffsetsW[FULLYCONNECTED_MAX_ELEMS];
    for (size_t i = 0; i < FULLYCONNECTED_MAX_ELEMS; i++) {
      gatherOffsetsA[i] = 0;
      gatherOffsetsW[i] = i * size;
    }
    int offsets[2] = { tAInput.getOffset(), tWInput.getOffset() };
    float scales[2] = { tAInput.getScale(), tWInput.getScale() };
    size_t mask = (1 << elems) - 1;
    __asm__ __volatile__(
        // Sets 1 lane enabled, moves scalar to float
        "mov.m.x     mt0, %[mask], 0\n"
        "flw.ps      f2, 0(%[sum])\n"            // Loads initial value
        "flw.ps      f3, 0(%[gatherOffsetsA])\n" // Loads gatherOffsets for gathers
        "flw.ps      f4, 0(%[gatherOffsetsW])\n" // Loads gatherOffsets for gathers
        // Loads scale and offset
        "fbc.ps      f5, 0x0(%[offsetA])\n"
        "fbc.ps      f6, 0x0(%[offsetW])\n"
        "fbc.ps      f7, 0x0(%[scaleA])\n"
        "fbc.ps      f8, 0x0(%[scaleW])\n"
        // Main loop
        "1:\n"
        "fgb.ps      f0, f3(%[tAAddr])\n"   // Loads data
        "fgb.ps      f1, f4(%[tWAddr])\n"
        "fsub.pi     f0, f0, f5\n"         // INT to FP32 dequantize: apply offset
        "fsub.pi     f1, f1, f6\n"
        "fcvt.ps.pw  f0, f0\n"             // INT to FP32 dequantize: convert to FP32
        "fcvt.ps.pw  f1, f1\n"
        "fmul.ps     f0, f0, f7\n"         // INT to FP32 dequantize: apply scale
        "fmul.ps     f1, f1, f8\n"
        "fmadd.ps    f2, f1, f0, f2\n"     // Accum
        // End of loop
        "addi   %[aCols], %[aCols], -1\n"
        "add    %[tAAddr], %[tAAddr], %[size]\n"         // Increment pointers
        "add    %[tWAddr], %[tWAddr], %[weightPitch]\n"
        "bne    %[aCols], x0, 1b\n"
        // Copies back to memory
        "fsw.ps f2, 0(%[sum])\n"
      : [tAAddr] "+&r" (tAAddr),
        [tWAddr] "+&r" (tWAddr),
        [aCols] "+&r" (aCols),
        [sum] "+&r" (sum)
      : [weightPitch] "r" (weightPitch),
        [mask] "r" (mask),
        [gatherOffsetsA] "r" (gatherOffsetsA),
        [gatherOffsetsW] "r" (gatherOffsetsW),
        [offsetA] "r" (&offsets[0]),
        [offsetW] "r" (&offsets[1]),
        [scaleA] "r" (&scales[0]),
        [scaleW] "r" (&scales[1]),
        [size] "r" (size)
      : "memory", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"
    );
  }
  // Others
  else {
    // For all the accumulations
    for (size_t aCol = 0; aCol < aCols; aCol++) {
      // Gets input value
      auto act = tAInput[actOffset + aCol];
      for (size_t elem = 0; elem < elems; elem++) {
        // Gets weight value
        auto weight = tWInput[weightOffset + elem + aCol * weightPitch];
        // Adds to the result
        sum[elem] += act * weight;
      }
    }
  }
}

/**
 * @brief Performs the FullyConnected operation between the activation, weights and bias.
 *
 * Executes a matrix multiply of the 2D tensor in in1T with the 2D tensor in
 * in2T. Each result is added with a bias specified in the 1D tensor in3T. The cols
 * of outT must be equal to the size of in3T. The results are stored in outT.
 * 
 * @tparam srcType Type of the elements of the tensors involved in the 
 *  FullyConnected (except for the bias)
 * @param[out] outT Tensor where we save the result of the convolution.
 * @param[in] in1T Tensor with the activations of the convolution.
 * @param[in] in2T Tensor with the weights of the convolution.
 * @param[in] in2T Tensor with the biases of the convolution.
 * @param[in] flags Controls the active shires and the type of evict that 
 *  should be done at the end of the function.
 */
template <ElemKind dstElK, ElemKind src3ElK>
inline void fwdLibFullyConnectedInst(LibTensor* outT, LibTensor* in1T,
                                     LibTensor* in2T, LibTensor* in3T,
                                     uint64_t flags,
                                     const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  
  using dstType  = typename elemKind2elemTy<dstElK>::type;
  using src1Type = typename elemKind2elemTy<dstElK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT --> dst  in1T--> inActT  in2T--> inWeighT in3T-->inBiasT */
  void *dstMatrix = outT->getRawDataPointer<void>();
  void *activations = in1T->getRawDataPointer<void>();
  void *weights = in2T->getRawDataPointer<void>();
  void *biases = nullptr;
  float biasScale = 1.0f;
  int32_t biasOffset = 0;
  if (in3T != nullptr) {
    biases = in3T->getRawDataPointer<void>();
    biasScale = in3T->getScale();
    biasOffset = in3T->getOffset();
  }
  
  Addresser<dstElK> tOutput(dstMatrix, outT->getScale(), outT->getOffset());
  const Addresser<dstElK> tAInput(activations, in1T->getScale(), in1T->getOffset());
  const Addresser<dstElK> tWInput(weights, in2T->getScale(), in2T->getOffset());
  const Addresser<src3ElK> tBias(biases, biasScale, biasOffset);
  
  const dim_t *dstIndex = outT->dims().data();
  const dim_t *actIndex = in1T->dims().data();
  
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *actPitch = in1T->strides().data();
  const dim_t *weightPitch = in2T->strides().data();
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = sizeof(dstType);
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstMatrix);
  if (maxRead == 0)
    return;

  unsigned int coord[2] = {0, 0};
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 2, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;

  bool done = false;
  // While there's work to do
  while (!done && (offsetOut < posMax)) {
    // 1.0 - Computes how many more elements can we compute
    size_t elems = FULLYCONNECTED_MAX_ELEMS;
    // 1.1 - Can't go beyond the elements left
    size_t elemsLeft = posMax - offsetOut;
    if(elems > elemsLeft) { elems = elemsLeft; }
    // 1.2 - Can't go beyond current row
    size_t colsLeft = dstIndex[1] - coord[1];
    if(elems > colsLeft) { elems = colsLeft; }

    typename accumulatorType<src1Type>::type sum[FULLYCONNECTED_MAX_ELEMS];
    // Starts the accumulation with the bias (per Channel)
    if (in3T != nullptr) {
      for (size_t i = 0; i < elems; i++) {
        sum[i] = tBias[coord[1] + i];
      }
    // No bias, start to 0
    } else {
      for (size_t i = 0; i < elems; i++) {
        sum[i] = 0;
      }
    }

    // Computes one result as efficient as possible
    matmulStep <dstElK> (sum, tAInput, activations, tWInput, weights, actIndex[1], coord[0] * actPitch[0], coord[1], weightPitch[0], elems);

    // Moves to next result
    for (size_t i = 0; i < elems; i++) {
      tOutput[offsetOut] = sum[i];
      done = getOffsets(2, coord, offsetOut, dstIndex, dstPitch);
    }
  }

  // Checks if evict is required
  if (DO_EVICTS) {
    unsigned int clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
    if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _FULLY_CONNECTED_INST_H_
