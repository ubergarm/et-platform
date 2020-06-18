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

#ifndef _MATMUL_INST_H_
#define _MATMUL_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
void fwdLibMatMulInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                      uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  //  using srcType = typename elemKind2elemTy<elK>::type;

  if (get_minion_id() != minionOffset) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  /* outT --> dst  in1--> act in2-> weight*/
  void* dst = outT->getRawDataPointer<void>();
  void* src = in1T->getRawDataPointer<void>();
  void* wei = in2T->getRawDataPointer<void>();
  
  // Addresser<elK> tOutput(dstMatrix, scale[2], offset[2]);
  Addresser<elK> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tAInput(activations, scale[0], offset[0]);
  const Addresser<elK> tAInput(src, in1T->getScale(), in1T->getOffset());
  // const Addresser<elK> tWInput(weights, scale[1], offset[1]);
  const Addresser<elK> tWInput(wei, in2T->getScale(), in2T->getOffset());

  //  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const dim_t *dstIndex = outT->dims().data();
  //unsigned int *actIndex = (unsigned int *)activationsDims;
  const dim_t *actIndex = in1T->dims().data();
  //unsigned int *weightIndex = (unsigned int *)weightsDims;
  // const dim_t * weightIndex = in2T->dims().data();

  //  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const dim_t *dstPitch = outT->strides().data();
  //unsigned int *actPitch = (unsigned int *)activationsPitches;
  const dim_t *actPitch = in1T->strides().data();
  //unsigned int *weightPitch = (unsigned int *)weightPitches;
  const dim_t *weightPitch = in2T->strides().data();
  
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

template <ElemKind elK>
void fwdLibMatMulInstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, 
                              uint64_t flags, const uint32_t minionOffset,
                              const uint32_t assignedMinions) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions)
    return;
  
  /* maintain compatibility through the new Iface Libtensor */
  /* outT --> dst  in1--> act in2-> weight*/
  void* dst = outT->getRawDataPointer<void>();
  void* src = in1T->getRawDataPointer<void>();
  void* wei = in2T->getRawDataPointer<void>();

  // Addresser<elK> tOutput(dstMatrix, scale[2], offset[2]);
  Addresser<elK> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tAInput(activations, scale[0], offset[0]);
  const Addresser<elK> tAInput(src, in1T->getScale(), in1T->getOffset());
  // const Addresser<elK> tWInput(weights, scale[1], offset[1]);
  const Addresser<elK>
    tWInput(wei, in2T->getScale(), in2T->getOffset());
  
  //  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const dim_t *dstIndex = outT->dims().data();
  //unsigned int *actIndex = (unsigned int *)activationsDims;
  const dim_t *actIndex = in1T->dims().data();
  //unsigned int *weightIndex = (unsigned int *)weightsDims;
  // const dim_t * weightIndex = in2T->dims().data();

  //  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const dim_t *dstPitch = outT->strides().data();
  //unsigned int *actPitch = (unsigned int *)activationsPitches;
  const dim_t *actPitch = in1T->strides().data();
  //unsigned int *weightPitch = (unsigned int *)weightPitches;
  const dim_t *weightPitch = in2T->strides().data();
  
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

  /* use overloading while sw-2400 and sw-2429 are WIP */
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++) {
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
    /* use overloading while sw-2400 and sw-2429 are WIP */
    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] == 0) {
      offsetAIn += actPitch[0];
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
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
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int regs, unsigned int extra, unsigned int length, unsigned int wgtStep, int32_t gatherValues[], const float *scale, const int32_t *offset){

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

    : [wgtAddr] "+&r" (wgtAddr),
      [dstAddr] "+&r" (dstAddr)
    : [regs]    "r" (regs),
      [extra]   "r" (extra),
      [wgtStep] "r" (wgtStep),
      [length]  "r" (length),
      [actAddr] "r" (actAddr)
    : "t0", "t1", "t2", "t3", "f0", "f1", "f2", "memory");

#undef MATMUL_ITERATION

}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int regs, unsigned int extra, unsigned int length, unsigned int wgtStep, int32_t gatherValues[], const float *scale, const int32_t *offset){

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

    : [wgtAddr] "+&r" (wgtAddr),
      [dstAddr] "+&r" (dstAddr)
    : [regs]    "r" (regs),
      [extra]   "r" (extra),
      [wgtStep] "r" (wgtStep),
      [length]  "r" (length),
      [actAddr] "r" (actAddr),
      [gthVals] "r" (gatherValues)
    : "t0", "t1", "t2", "t3", "f0", "f1", "f2", "f31", "memory");

#undef MATMUL_ITERATION

}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int regs, unsigned int extra, unsigned int length, unsigned int wgtStep, int32_t gatherValues[], const float *scale, const int32_t *offset){

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
      INT8_TO_FP32(f0, f16, f17)                   \
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
    "fbc.ps   f16, 0x0(%[scale]) \n"
    "fbc.ps   f17, 0x0(%[offset]) \n"
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

    : [wgtAddr] "+&r" (wgtAddr),
      [dstAddr] "+&r" (dstAddr)
    : [regs]    "r" (regs),
      [extra]   "r" (extra),
      [wgtStep] "r" (wgtStep),
      [length]  "r" (length),
      [actAddr] "r" (actAddr),
      [gthVals] "r" (gatherValues),
      [scale]   "r" (scale),
      [offset]  "r" (offset)
    : "t0", "t1", "t2", "t3", "f0", "f1", "f2", "f3", "f4", "f16", "f17", "f28", "f29", "f30", "f31", "memory");

#undef MATMUL_ITERATION
#undef INT8_TO_FP32
#undef FP32_TO_INT8
#undef SIGN_EXTEND_INT8_REG
}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value && !std::is_same<srcType, float16>::value && !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int regs, unsigned int extra, unsigned int length, unsigned int wgtStep, int32_t gatherValues[], const float *scale, const int32_t *offset){}

// 2-D MATRIX MULTIPLICATION: THREADED AND VECTORIZED VERSION.
// Assumption: there is no padding in the last dimension (in this case, the second one),
// since all the tensors involved are 2D. It is necessary to assume this at least for the
// wgt and dst tensors, so we can use vectorization.

template <ElemKind elK>
void fwdLibMatMulInstVectorized(LibTensor* outT, LibTensor* in1T,
                                LibTensor* in2T, uint64_t flags,
                                const uint32_t minionOffset,
                                const uint32_t assignedMinions) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT --> dst  in1--> act in2-> weight*/
  void* dst = outT->getRawDataPointer<void>();
  void* src = in1T->getRawDataPointer<void>();
  void* wei = in2T->getRawDataPointer<void>();

  // unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const dim_t *actIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const dim_t *actPitch = in1T->strides().data();
  // unsigned int *wgtPitch = (unsigned int *)weightPitches;
  const dim_t *wgtPitch = in2T->strides().data();
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int posMax = initialAddr + maxRead;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int last_non_zero_coord = 0;

  /* use overloading while sw-2400 and sw-2429 are WIP */
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex,
                           last_non_zero_coord);

// The vector coordOut now contains the coordinates for the first position in the dst tensor that should be written by the minion.
// Additionally, posMax indicates the last position in the dst tensor that should be written by the minion, plus one.

  unsigned int offsetOut = 0; // Position in destination tensor.
  for (unsigned int i = 0; i < last_non_zero_coord; i++) {
    offsetOut += coordOut[i] * dstPitch[i];
  }
  unsigned int offsetAIn = coordOut[0]*actPitch[0]; // Position in activation tensor.
  unsigned int offsetWIn = coordOut[1]; // Position in weight tensor.

// Once the tensor offsets have been obtained, comes the vectorization part.
// The following vector property is fundamental to understand this implementation.
// Consider the product A x W = D. Let a_x denote the xth row of the tensor A, and
// a_xy denote the yth element of the xth row of the tensor A (and the same for the
// other tensors). Then, d_x = sum{a_xy*w_y}, where the sum is performed for each y.
// In words: the xth row of the destination tensor is obtained by summing each weight
// row multiplied by the corresponding element of the activation row a_x.

// The following code uses a function named MatMulOp, which computes a whole row of
// the destination tensor using the property above. The following lines compute
// the parameters needed so the product can be performed.

  unsigned int currentRow = coordOut[0];
  unsigned int lastRow = posMax/dstPitch[0]; // Row corresponding to the posMax position in the dst tensor.
  if (posMax%dstPitch[0] > dstIndex[1]) posMax = lastRow*dstPitch[0] + dstIndex[1];
  unsigned int lastElems = posMax - dstPitch[0]*lastRow; // Number of elements covered by the minion in its last row. Could be zero.

  unsigned int firstRowElems, regsPerRow, extraPerRow;
  if (currentRow == lastRow) firstRowElems = posMax - offsetOut;
  else firstRowElems = dstIndex[1] - offsetOut%dstPitch[0];
  regsPerRow = dstIndex[1]/8;
  extraPerRow = dstIndex[1] - 8*regsPerRow;
  
  unsigned int regs = firstRowElems/8;
  unsigned int extra = firstRowElems - 8*regs;

  unsigned int length = actIndex[1]; // Length of the dot products that will be performed.
  unsigned int wgtStep = wgtPitch[0]*typeSize;

  int32_t gatherValues[8];
  setGatherValues <srcType>(gatherValues);

  float scale[3] =  {in1T->getScale(), in2T->getScale(), outT->getScale()};
  int32_t offset[3] = {in1T->getOffset(), in2T->getOffset(), outT->getOffset()};
  while (true) {
    // uintptr_t dstAddr = (uintptr_t)dstMatrix + typeSize*offsetOut;
    // uintptr_t actAddr = (uintptr_t)activations + typeSize*offsetAIn;
    // uintptr_t wgtAddr = (uintptr_t)weights + typeSize*offsetWIn;
    uintptr_t dstAddr = (uintptr_t)dst + typeSize*offsetOut;
    uintptr_t actAddr = (uintptr_t)src + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)wei + typeSize*offsetWIn;
     
    matmulOp <srcType>(dstAddr, actAddr, wgtAddr, regs, extra, length, wgtStep, gatherValues, scale, offset);

    ++currentRow;
    if (currentRow > lastRow) break; // The loop ends when all the minion rows have been covered.

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
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


} // namespace inlining

} // namespace dnn_lib

#endif // _MATMUL_INST_H_
