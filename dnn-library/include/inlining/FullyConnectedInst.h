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

template <ElemKind dstElK, ElemKind src1ElK, ElemKind src2ElK>
inline void fwdLibFullyConnectedInst(LibTensor* outT, LibTensor* in1T,
                                     LibTensor* in2T, LibTensor* in3T) {

  using dstType  = typename elemKind2elemTy<dstElK>::type;
  using src1Type = typename elemKind2elemTy<src1ElK>::type;
  using src2Type = typename elemKind2elemTy<src2ElK>::type;

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT --> dst  in1T--> inActT  in2T--> inWeighT in3T-->inBiasT */

  void *dstMatrix = outT->getRawDataPointer<void>();
  void *activations = in1T->getRawDataPointer<void>();
  void *weights = in2T->getRawDataPointer<void>();
  // float *tBias = (float *)bias;
  float *tBias = in3T->getRawDataPointer<float>();
  
  // Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  Addresser<dstType> tOutput(dstMatrix, outT->getScale(), outT->getOffset());
  const Addresser<src1Type> tAInput(activations, in1T->getScale(), in1T->getOffset());
  const Addresser<src2Type> tWInput(weights, in2T->getScale(), in2T->getOffset());

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
  
  // For each (x,y) in the destination matrix:
  for (unsigned int x = 0; x < dstIndex[0]; x++) {
    for (unsigned int y = 0; y < dstIndex[1]; y++) {
      // Perform DOT on the row an column.
      float sum = 0.0;
      for (unsigned int i = 0; i < actIndex[1]; i++) {
        sum += float(tAInput[x * actPitch[0] + i] *
                     tWInput[i * weightPitch[0] + y]);
      }
      sum += tBias[y];
      tOutput[x * dstPitch[0] + y] = sum;
    }
  }
}

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
template <typename srcType,
          typename std::enable_if<std::is_same<srcType, int64_t>::value, std::size_t>::type = 0>
inline void matmulStep (int64_t *sum,
                        const Addresser<srcType> &tAInput,
                        void * tAInputPtr,
                        const Addresser<srcType> &tWInput,
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
template <typename srcType,
          typename std::enable_if<std::is_same<srcType, int32_t>::value, std::size_t>::type = 0>
inline void matmulStep (int32_t *sum,
                        const Addresser<srcType> &tAInput,
                        void * tAInputPtr,
                        const Addresser<srcType> &tWInput,
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
template <typename srcType>
inline void matmulStep (float *sum,
                        const Addresser<srcType> &tAInput,
                        void * tAInputPtr,
                        const Addresser<srcType> &tWInput,
                        void * tWInputPtr,
                        size_t aCols,
                        size_t actOffset,
                        size_t weightOffset,
                        size_t weightPitch,
                        size_t elems) {

  // Float version
  if (std::is_same<srcType, float>::value) {
    char * tAAddr = (char *) tAInputPtr;
    tAAddr += actOffset * 4;
    char * tWAddr = (char *) tWInputPtr;
    tWAddr += weightOffset * 4;
    weightPitch *= 4;
    int32_t offsets[FULLYCONNECTED_MAX_ELEMS];
    for (size_t i = 0; i < FULLYCONNECTED_MAX_ELEMS; i++) {
      offsets[i] = i * 4;
    }
    size_t mask = (1 << elems) - 1;
    __asm__ __volatile__(
        // Sets 1 lane enabled, moves scalar to float
        "mov.m.x	   mt0, %[mask], 0\n"
        "flw.ps      f2, 0(%[sum])\n"     // Loads initial value
        "flw.ps      f3, 0(%[offsets])\n" // Loads offsets for gathers
        // Main loop
        "1:\n"
        "fbc.ps      f0, 0(%[tAAddr])\n"   // Loads data
        "flw.ps      f1, 0(%[tWAddr])\n"
        "fmadd.ps    f2, f1, f0, f2\n"     // Accum
        // End of loop
        "addi   %[aCols], %[aCols], -1\n"
        "addi   %[tAAddr], %[tAAddr], 4\n"              // Increment pointers
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
        [offsets] "r" (offsets)
      : "memory", "f0", "f1", "f2", "f3"
    );
  }
  // Float16 version
  else if (std::is_same<srcType, float16>::value) {
    char * tAAddr = (char *) tAInputPtr;
    tAAddr += actOffset * 2;
    char * tWAddr = (char *) tWInputPtr;
    tWAddr += weightOffset * 2;
    weightPitch *= 2;
    int32_t offsets[FULLYCONNECTED_MAX_ELEMS];
    for (size_t i = 0; i < FULLYCONNECTED_MAX_ELEMS; i++) {
      offsets[i] = i * 2;
    }
    size_t mask = (1 << elems) - 1;
    __asm__ __volatile__(
        // Sets 1 lane enabled, moves scalar to float
        "mov.m.x	   mt0, %[mask], 0\n"
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
        "addi   %[tAAddr], %[tAAddr], 2\n"              // Increment pointers
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
        [offsets] "r" (offsets)
      : "memory", "f0", "f1", "f2", "f3"
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
template <ElemKind dstElK, ElemKind src1ElK, ElemKind src2ElK>
inline void fwdLibFullyConnectedInstThreaded(LibTensor* outT, LibTensor* in1T,
                                             LibTensor* in2T, LibTensor* in3T,
                                             uint64_t flags) {
  
  using dstType  = typename elemKind2elemTy<dstElK>::type;
  using src1Type = typename elemKind2elemTy<src1ElK>::type;
  using src2Type = typename elemKind2elemTy<src2ElK>::type;

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT --> dst  in1T--> inActT  in2T--> inWeighT in3T-->inBiasT */
  void *dstMatrix = outT->getRawDataPointer<void>();
  void *activations = in1T->getRawDataPointer<void>();
  void *weights = in2T->getRawDataPointer<void>();
  float *tBias = in3T->getRawDataPointer<float>();
  
  Addresser<dstType> tOutput(dstMatrix, outT->getScale(), outT->getOffset());
  const Addresser<src1Type> tAInput(activations, in1T->getScale(), in1T->getOffset());
  const Addresser<src2Type> tWInput(weights, in2T->getScale(), in2T->getOffset());
  
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

    // Starts the accumulation with the bias (per Channel)
    typename accumulatorType<src1Type>::type sum[FULLYCONNECTED_MAX_ELEMS];
    for (size_t i = 0; i < elems; i++) {
      sum[i] = tBias[coord[1] + i];
    }

    // Computes one result as efficient as possible
    matmulStep <src1Type> (sum, tAInput, activations, tWInput, weights, actIndex[1], coord[0] * actPitch[0], coord[1], weightPitch[0], elems);

    // Moves to next result
    for (size_t i = 0; i < elems; i++) {
      tOutput[offsetOut] = sum[i];
      done = getOffsets(2, coord, offsetOut, dstIndex, dstPitch);
    }
  }

  // Checks if evict is required
  if (DO_EVICTS) {
    unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
    if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
  }
}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, float>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset){

#define MATMUL_ITERATION               \
    "flw.ps   f0, 0x0(%[actAddr])\n"   \
    "fgw.ps   f1, f29(%[wgtAddr])\n"   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x20\n"
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"

    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f30, 0x0(%[biasAddr])\n"
    "fadd.s f31, f30, f31\n"
    "fsw.ps f31, 0x0(%[dstAddr])\n"

    :
    : [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr)
    : "t0", "t1", "f0", "f1", "f29", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, float16>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset){

#define MATMUL_ITERATION               \
    "fgh.ps   f0, f28(%[actAddr])\n"   \
    "fgh.ps   f1, f29(%[wgtAddr])\n"   \
    "fcvt.ps.f16 f0, f0 \n"            \
    "fcvt.ps.f16 f1, f1 \n"            \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f28, 0x0(%[gthValuesAct])\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x10\n"
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"

    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f30, 0x0(%[biasAddr])\n"
    "fadd.s f31, f30, f31\n"
    "fcvt.f16.ps f31, f31\n"           // Conversion fp32 >> fp16.
    "fsch.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr)
    : "t0", "t1", "f0", "f1", "f28", "f29", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset){

#define INT8_TO_FP32(_reg)                  \
    "fsub.pi " #_reg ", " #_reg ", f16 \n"  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"    \
    "fmul.ps " #_reg ", " #_reg ", f17 \n"

#define MATMUL_ITERATION               \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f16, 0x0(%[offset]) \n"  \
    "fbc.ps   f17, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                   \
    "fbc.ps   f16, 0x4(%[offset]) \n"  \
    "fbc.ps   f17, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define FP32_TO_INT8(_reg)                      \
    "frcp.ps f17, f17 \n"                       \
    "fcvt.ps.pw f16, f16 \n"                    \
    "fmadd.ps " #_reg ", " #_reg ", f17, f16 \n" \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsat8.pi " #_reg ", " #_reg "\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f28, 0x0(%[gthValuesAct])\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x8\n"
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"

    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f30, 0x0(%[biasAddr])\n"
    "fadd.s f31, f30, f31\n"
    "fbc.ps f16, 0x8(%[offset]) \n"
    "fbc.ps f17, 0x8(%[scale]) \n"
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f16", "f17", "f28", "f29", "f30", "f31", "memory");

#undef INT8_TO_FP32
#undef MATMUL_ITERATION
#undef FP32_TO_INT8
}

#define INT8_TO_FP32(_reg)                  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"    \
    "fmul.ps " #_reg ", " #_reg ", f17 \n"

#define UINT8_TO_FP32(_reg)                   \
    "fandi.pi " #_reg ", " #_reg ", 0xff \n"  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"      \
    "fmul.ps " #_reg ", " #_reg ", f17 \n"


#define MATMUL_ITERATION_U8_U8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f17, 0x0(%[scale]) \n"   \
    UINT8_TO_FP32(f0)                   \
    "fbc.ps   f17, 0x4(%[scale]) \n"   \
    UINT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define MATMUL_ITERATION_I8_U8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f17, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                   \
    "fbc.ps   f17, 0x4(%[scale]) \n"   \
    UINT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define MATMUL_ITERATION_U8_I8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f17, 0x0(%[scale]) \n"   \
    UINT8_TO_FP32(f0)                   \
    "fbc.ps   f17, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define MATMUL_ITERATION_I8_I8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f17, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                   \
    "fbc.ps   f17, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"


#define FP32_TO_INT8(_reg)                      \
    "frcp.ps f17, f17 \n"                       \
    "fmadd.ps " #_reg ", " #_reg ", f17, f16 \n" \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsat8.pi " #_reg ", " #_reg "\n"


#define FP32_TO_UINT8(_reg)                     \
    "frcp.ps f17, f17 \n"                       \
    "fmul.ps " #_reg ", " #_reg ", f17 \n"      \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsrli.pi f2," #_reg ", 0x8 \n"             \
    "fxor.pi f17, f17, f17 \n"                  \
    "fcmov.ps " #_reg" , f12, f17, " #_reg " \n"


#define STEP1                                            \
    "mov.m.x m0, zero, 0xff\n"                           \
    "xor t0, t0, t0\n"                                   \
    "flw.ps f28, 0x0(%[gthValuesAct])\n"                 \
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"                 \
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"                   \
    "fxor.pi f31, f31, f31\n"                            \
                                                         \
    "1:\n"                                               \
    "addi     t0, t0, 8\n"                               \
    "ble      %[elemsRow], t0, 2f\n"

#define STEP2                                            \
    "addi %[actAddr], %[actAddr], 0x8\n"                 \
    "fadd.pi f29, f29, f30\n"                            \
    "beq      zero, zero, 1b\n"                          \
    "2:\n"                                               \
    "fxor.pi  f0, f0, f0\n"                              \
    "addi     t0, t0, -8\n"                              \
    "sub      t0, %[elemsRow], t0\n"                     \
    "addi     t1, zero, 1\n"                             \
    "sll      t1, t1, t0\n"                              \
    "addi     t1, t1, -1\n"                              \
    "mov.m.x  m0, t1, 0\n"                               \

#define STEP3                                            \
    "fmvs.x.ps t0, f31, 0x4\n"                           \
    "fmv.w.x   f30, t0\n"                                \
    "fadd.s    f31, f30, f31\n"                          \
    "mov.m.x m0, zero, 0x1\n"                            \
    "fbc.ps f30, 0x0(%[biasAddr])\n"                     \
    "fadd.s f31, f30, f31\n"                             \
    "fbc.ps f16, 0x8(%[offset]) \n"                      \
    "fbc.ps f17, 0x8(%[scale]) \n"                       \


template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_I8
    STEP2
    MATMUL_ITERATION_U8_I8
    STEP3
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f16", "f17", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset){
  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_I8_U8
    STEP2
    MATMUL_ITERATION_I8_U8
    STEP3
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f16", "f17", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_I8_I8
    STEP2
    MATMUL_ITERATION_I8_I8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f16", "f17", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_I8_U8
    STEP2
    MATMUL_ITERATION_I8_U8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f16", "f17", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_I8
    STEP2
    MATMUL_ITERATION_U8_I8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f16", "f17", "f28", "f29", "f30", "f31", "memory");


}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset) {

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_U8
    STEP2
    MATMUL_ITERATION_U8_U8
    STEP3
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f16", "f17", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_U8
    STEP2
    MATMUL_ITERATION_U8_U8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f16", "f17", "f28", "f29", "f30", "f31", "memory");

}

#undef INT8_TO_FP32
#undef UINT8_TO_FP32
#undef MATMUL_ITERATION_U8_U8
#undef MATMUL_ITERATION_I8_U8
#undef MATMUL_ITERATION_U8_I8
#undef MATMUL_ITERATION_I8_I8
#undef FP32_TO_INT8
#undef FP32_TO_UINT8
#undef STEP1
#undef STEP2
#undef STEP3

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<!std::is_same<src1Type, int8_t>::value && !std::is_same<src1Type, float16>::value && !std::is_same<src1Type, float>::value && !std::is_same<src1Type, uint8_t>::value, std::size_t>::type = 0>
inline void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, const float *scale, const int32_t *offset){}

template <ElemKind dstElK, ElemKind src1ElK, ElemKind src2ElK>
inline void fwdLibFullyConnectedInstVectorized(LibTensor* outT, LibTensor* in1T,
                                               LibTensor* in2T, LibTensor* in3T,
                                               const float* scale,
                                               const int32_t* offset,
                                               uint64_t flags) {
  
  using dstType  = typename elemKind2elemTy<dstElK>::type;
  using src1Type = typename elemKind2elemTy<src1ElK>::type;
  using src2Type = typename elemKind2elemTy<src2ElK>::type;

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT --> dst  in1T--> inActT  in2T--> inWeighT in3T-->inBiasT */
  void *dstMatrix = outT->getRawDataPointer<void>();
  void *activations = in1T->getRawDataPointer<void>();
  void *weights = in2T->getRawDataPointer<void>();
  float *bias = in3T->getRawDataPointer<float>();

  // unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const dim_t *actIndex = in1T->dims().data();
  
  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches1;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const dim_t *actPitch = in1T->strides().data();
  // unsigned int *weightPitch = (unsigned int *)weightPitches;
  const dim_t *weightPitch = in2T->strides().data();
  
  // Total number of elements to process is the size of the outter
  // dimension of the destination tensor multiplied by its pitch
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<src1Type>();
  
  // Gets the total number of elements to work on for the minion
  // initialAddr: is first element to start working on
  // maxRead: number of elements to process
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[2];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 2, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  unsigned int offsetAIn = coord[0]*actPitch[0];

  int32_t gatherValuesAct[8], gatherValuesWgt[8];
  gatherValuesAct[0] = gatherValuesWgt[0] = 0;
  unsigned int step = weightPitch[0]*typeSize;
  for (unsigned int i = 1; i < 8; ++i) {
    gatherValuesAct[i] = gatherValuesAct[i - 1] + typeSize;
    gatherValuesWgt[i] = gatherValuesWgt[i - 1] + step;
  }
  unsigned int wgtRegStep = 8*step;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    uintptr_t dstAddr = (uintptr_t)dstMatrix + typeSize*offsetOut;
    uintptr_t actAddr = (uintptr_t)activations + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)weights + typeSize*coord[1];
    uintptr_t biasAddr = (uintptr_t)bias + 4*coord[1]; // bias is a float vector.
    fullyConnectedOp <src1Type, src2Type, dstType>(dstAddr, actAddr, wgtAddr, actIndex[1], gatherValuesAct,
                       gatherValuesWgt, wgtRegStep, biasAddr, scale, offset);
    done = getOffsets(2, coord, offsetOut, dstIndex, dstPitch);
    if (coord[1] == 0) {
      offsetAIn += actPitch[0];
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _FULLY_CONNECTED_INST_H_
