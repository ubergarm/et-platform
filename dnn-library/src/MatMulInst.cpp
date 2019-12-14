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

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "LibNodes.h"
#include "GenInstances.h"
#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"

using namespace std;

template <typename srcType>
void dnn_lib::fwdLibMatMulInst(void *dstMatrix, void *dstMatrixDims,
                               void *dstMatrixPitches, void *activations,
                               void *activationsDims, void *activationsPitches,
                               void *weights, void *weightsDims,
                               void *weightPitches, float *scale,
                               int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[2], offset[2]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  // For each (x,y) in the destination matrix:
  for (unsigned int x = 0; x < dstIndex[0]; x++) {
    for (unsigned int y = 0; y < dstIndex[1]; y++) {
      // Perform DOT on the row an column.
      float sum = 0;
      for (unsigned int i = 0; i < actIndex[1]; i++) {
        sum += float(tAInput[x * actPitch[0] + i]) *
               float(tWInput[i * weightPitch[0] + y]);
      }
      tOutput[x * dstPitch[0] + y] = float(sum);
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibMatMulInstThreaded(void *dstMatrix, void *dstMatrixDims,
                                       void *dstMatrixPitches,
                                       void *activations, void *activationsDims,
                                       void *activationsPitches, void *weights,
                                       void *weightsDims, void *weightPitches,
                                       float *scale, int32_t *offset,
                                       uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[2], offset[2]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int k = 0;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coordOut[i] * dstPitch[i];
  }
  unsigned int offsetAIn = coordOut[0]*actPitch[0];

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    float sum = 0;
    unsigned int weightOffset = 0;
    for (unsigned int i = 0; i < actIndex[1]; i++) {
      sum += float(tAInput[offsetAIn + i]) *
             float(tWInput[weightOffset + coordOut[1]]);
      weightOffset += weightPitch[0];
    }
    tOutput[offsetOut] = float(sum);
    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] == 0) {
      offsetAIn += actPitch[0];
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, std::size_t>::type = 0>
void setGatherValues (int32_t gatherValues[]){
  gatherValues[0] = 0;
  for (unsigned int i = 1; i < 8; ++i) gatherValues[i] = gatherValues[i - 1] + 2;
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value, std::size_t>::type = 0>
void setGatherValues (int32_t gatherValues[]){
  gatherValues[0] = 0;
  for (unsigned int i = 1; i < 8; ++i) gatherValues[i] = gatherValues[i - 1] + 1;
}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value && !std::is_same<srcType, float16>::value, std::size_t>::type = 0>
void setGatherValues (int32_t gatherValues[]){} // includes the case srcType = float.

template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int regs, unsigned int extra, unsigned int length, unsigned int wgtStep, int32_t gatherValues[], float *scale, int32_t *offset){

#define MATMUL_ITERATION                           \
    "fxor.pi f2, f2, f2\n"                         \
    "xor t1, t1, t1\n"                             \
    "add t2, %[actAddr], zero\n"                   \
    "add t3, %[wgtAddr], zero\n"                   \
    "4:\n"                                         \
    "addi t1, t1, 0x1\n"                           \
    "blt  %[length], t1, 5f\n"                     \
      "fbc.ps f0, 0x0(t2)\n"                       \
      "flw.ps f1, 0x0(t3)\n"                       \
      "fmadd.ps f2, f0, f1, f2\n"                  \
      "addi t2, t2, 0x4\n"                         \
      "add  t3, t3, %[wgtStep]\n"                  \
    "j 4b\n"                                       \
    "5:\n"                                         \
    "fsw.ps f2, 0x0(%[dstAddr])\n"                 \
    "addi %[wgtAddr], %[wgtAddr], 0x20 \n"         \
    "addi %[dstAddr], %[dstAddr], 0x20 \n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"

    "1:\n"
    "addi     t0, t0, 0x1\n"
    "blt      %[regs], t0, 2f\n"
    MATMUL_ITERATION
    "j 1b\n"
                                         
    "2:\n"
    "beq %[extra], zero, 3f\n"
    "addi     t0, zero, 1\n"
    "sll      t0, t0, %[extra]\n"
    "addi     t0, t0, -1\n"
    "mov.m.x  m0, t0, 0\n"
    MATMUL_ITERATION

    "3:\n"

    :
    : [regs]    "r" (regs),
      [extra]   "r" (extra),
      [wgtStep] "r" (wgtStep),
      [length]  "r" (length),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr)
    : "t0", "t1", "f0", "f1", "f2", "memory");

#undef MATMUL_ITERATION

}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int regs, unsigned int extra, unsigned int length, unsigned int wgtStep, int32_t gatherValues[], float *scale, int32_t *offset){

#define MATMUL_ITERATION                           \
    "fxor.pi f2, f2, f2\n"                         \
    "xor t1, t1, t1\n"                             \
    "add t2, %[actAddr], zero\n"                   \
    "add t3, %[wgtAddr], zero\n"                   \
    "4:\n"                                         \
    "addi t1, t1, 0x1\n"                           \
    "blt  %[length], t1, 5f\n"                     \
      "fbc.ps f0, 0x0(t2)\n"                       \
      "fcvt.ps.f16 f0, f0 \n"                      \
      "fgh.ps f1, f31(t3)\n"                       \
      "fcvt.ps.f16 f1, f1 \n"                      \
      "fmadd.ps f2, f0, f1, f2\n"                  \
      "addi t2, t2, 0x2\n"                         \
      "add  t3, t3, %[wgtStep]\n"                  \
    "j 4b\n"                                       \
    "5:\n"                                         \
    "fcvt.f16.ps f2, f2\n"                         \
    "fsch.ps f2, f31(%[dstAddr])\n"                \
    "addi %[wgtAddr], %[wgtAddr], 0x10 \n"         \
    "addi %[dstAddr], %[dstAddr], 0x10 \n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "flw.ps f31, 0x0(%[gthVals])\n"
    "xor t0, t0, t0\n"

    "1:\n"
    "addi     t0, t0, 0x1\n"
    "blt      %[regs], t0, 2f\n"
    MATMUL_ITERATION
    "j 1b\n"
                                         
    "2:\n"
    "beq %[extra], zero, 3f\n"
    "addi     t0, zero, 1\n"
    "sll      t0, t0, %[extra]\n"
    "addi     t0, t0, -1\n"
    "mov.m.x  m0, t0, 0\n"
    MATMUL_ITERATION

    "3:\n"

    :
    : [regs]    "r" (regs),
      [extra]   "r" (extra),
      [wgtStep] "r" (wgtStep),
      [length]  "r" (length),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [gthVals] "r" (gatherValues)
    : "t0", "t1", "f0", "f1", "f2", "memory");

#undef MATMUL_ITERATION

}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int regs, unsigned int extra, unsigned int length, unsigned int wgtStep, int32_t gatherValues[], float *scale, int32_t *offset){

#define INT8_TO_FP32(_reg, _scl, _off)            \
    "fsub.pi " #_reg ", " #_reg ", " #_off " \n"  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"          \
    "fmul.ps " #_reg ", " #_reg ", " #_scl " \n"

// Assuming the inverse scale is in f30 and the offset in f31.
#define FP32_TO_INT8(_reg)                        \
    "fmul.ps " #_reg ", " #_reg ", f30\n"         \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"           \
    "fadd.pi " #_reg ", " #_reg ", f31\n"         \
    "fsat8.pi " #_reg ", " #_reg "\n"

// Sign extension of an int8 registger, using f4 as auxilliary register. Non-optimal.
#define SIGN_EXTEND_INT8_REG(_reg)                 \
      "fandi.pi " #_reg ", " #_reg ", 0xff\n"      \
      "fandi.pi f4, " #_reg ", 0x7f\n"             \
      "feq.pi   f4, " #_reg ", f4\n"               \
      "fnot.pi  f4, f4\n"                          \
      "fandi.pi f4, f4, -256\n"                    \
      "fadd.pi  " #_reg ", " #_reg ", f4\n"

#define MATMUL_ITERATION                           \
    "fxor.pi f2, f2, f2\n"                         \
    "xor t1, t1, t1\n"                             \
    "add t2, %[actAddr], zero\n"                   \
    "add t3, %[wgtAddr], zero\n"                   \
    "4:\n"                                         \
    "addi t1, t1, 0x1\n"                           \
    "blt  %[length], t1, 5f\n"                     \
      "fbc.ps f0, 0x0(t2)\n"                       \
      SIGN_EXTEND_INT8_REG(f0)                     \
      INT8_TO_FP32(f0, f26, f27)                   \
      "fgb.ps f1, f3(t3)\n"                        \
      INT8_TO_FP32(f1, f28, f29)                   \
      "fmadd.ps f2, f0, f1, f2\n"                  \
      "addi t2, t2, 0x1\n"                         \
      "add  t3, t3, %[wgtStep]\n"                  \
    "j 4b\n"                                       \
    "5:\n"                                         \
    FP32_TO_INT8(f2)                               \
    "fscb.ps f2, f3(%[dstAddr])\n"                 \
    "addi %[wgtAddr], %[wgtAddr], 0x8 \n"          \
    "addi %[dstAddr], %[dstAddr], 0x8 \n"

   __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "flw.ps f3, 0x0(%[gthVals])\n"
    "xor t0, t0, t0\n"

    // Activation tensor scale and offset
    "fbc.ps   f26, 0x0(%[scale]) \n"
    "fbc.ps   f27, 0x0(%[offset]) \n"
    // Weight tensor scale and offset
    "fbc.ps   f28, 0x4(%[scale]) \n"
    "fbc.ps   f29, 0x4(%[offset]) \n"
    // Output tensor inverse scale and offset.
    "fbc.ps   f30, 0x8(%[scale]) \n"
    "frcp.ps f30, f30 \n"
    "fbc.ps   f31, 0x8(%[offset]) \n"

    "1:\n"
    "addi     t0, t0, 0x1\n"
    "blt      %[regs], t0, 2f\n"
    MATMUL_ITERATION
    "j 1b\n"
                                         
    "2:\n"
    "beq %[extra], zero, 3f\n"
    "addi     t0, zero, 1\n"
    "sll      t0, t0, %[extra]\n"
    "addi     t0, t0, -1\n"
    "mov.m.x  m0, t0, 0\n"
    MATMUL_ITERATION

    "3:\n"

    :
    : [regs]    "r" (regs),
      [extra]   "r" (extra),
      [wgtStep] "r" (wgtStep),
      [length]  "r" (length),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [gthVals] "r" (gatherValues),
      [scale]   "r" (scale),
      [offset]  "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f3", "f4", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

#undef MATMUL_ITERATION
#undef INT8_TO_FP32
#undef FP32_TO_INT8
#undef SIGN_EXTEND_INT8_REG
}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value && !std::is_same<srcType, float16>::value && !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int regs, unsigned int extra, unsigned int length, unsigned int wgtStep, int32_t gatherValues[], float *scale, int32_t *offset){}

// Default version of MatMul in case the input weights have not been previously transposed.
// Assumption: there is no padding in the last dimension (in this case, the second one),
// since all the tensors involved are 2D. It is necessary to assume this at least for the
// wgt and dst tensors, so we can use vectorization.
template <typename srcType>
void dnn_lib::fwdLibMatMulInstVectorized(void *dstMatrix, void *dstMatrixDims,
                                         void *dstMatrixPitches,
                                         void *activations, void *activationsDims,
                                         void *activationsPitches, void *weights,
                                         void *weightsDims, void *weightPitches,
                                         float *scale, int32_t *offset, uint64_t flags,
                                         const uint32_t minionOffset,
                                         const uint32_t assignedMinions) {

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (32 * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions)
    return;

// For debugging
  Addresser<srcType> tOutput(dstMatrix, scale[4], offset[4]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *wgtPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions); // Output: initialAddr, maxRead.
  if (maxRead == 0)
    return;

  unsigned int posMax = initialAddr + maxRead;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int last_non_zero_coord = 0;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, last_non_zero_coord); // Output: coordOut.

  unsigned int offsetOut = 0;
  for (int i = 0; i < last_non_zero_coord; i++) {
    offsetOut += coordOut[i] * dstPitch[i];
  }
  unsigned int offsetAIn = coordOut[0]*actPitch[0];
  unsigned int offsetWIn = coordOut[1];

  unsigned int currentRow = coordOut[0];
  unsigned int lastRow = posMax/dstPitch[0]; // The number of elements to cover in the last row could be zero.
  if (posMax%dstPitch[0] > dstIndex[1]) posMax = lastRow*dstPitch[0] + dstIndex[1];
  unsigned int lastElems = posMax - dstPitch[0]*lastRow; // Number of elements covered by the minion in its last row. Could be zero.

  unsigned int firstRowElems, regsPerRow, extraPerRow;
  if (currentRow == lastRow) firstRowElems = posMax - offsetOut;
  else {
    firstRowElems = dstIndex[1] - offsetOut%dstPitch[0];
    regsPerRow = dstIndex[1]/8;
    extraPerRow = dstIndex[1] - 8*regsPerRow;
  }
  unsigned int regs = firstRowElems/8;
  unsigned int extra = firstRowElems - 8*regs;
  unsigned int length = actIndex[1]; // Length of the dot products that will be performed.
  unsigned int wgtStep = wgtPitch[0]*typeSize;

  int32_t gatherValues[8];
  setGatherValues <srcType>(gatherValues);

  while (true) {
    uintptr_t dstAddr = (uintptr_t)dstMatrix + typeSize*offsetOut;
    uintptr_t actAddr = (uintptr_t)activations + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)weights + typeSize*offsetWIn;
    matmulOp <srcType>(dstAddr, actAddr, wgtAddr, regs, extra, length, wgtStep, gatherValues, scale, offset);

    ++currentRow;
    if (currentRow > lastRow) break;

    // Updating of tensor offsets: offsetWIn, offsetAIn, offsetOut.
    offsetWIn = 0;
    offsetAIn += actPitch[0];
    offsetOut += dstPitch[0] - coordOut[1];
    coordOut[1] = 0;
    // Updating of the number of 8-lane registers and extra lanes in the current dst tensor row.
    if (currentRow < lastRow) {
      regs = regsPerRow;
      extra = extraPerRow;
    }
    else {
      regs = lastElems/8;
      extra = lastElems - 8*regs;
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}


GEN_INSTANCES_OP(template, fwdLibMatMulInst, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibMatMulInstThreaded, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibMatMulInstVectorized, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         float *scale, int32_t *offset, uint64_t flags,
                         const uint32_t minionOffset = 0, const uint32_t numShires = 0);
