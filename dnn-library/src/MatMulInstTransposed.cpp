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
void dnn_lib::fwdLibMatMulInstTransposed(void *dstMatrix, void *dstMatrixDims,
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
               float(tWInput[y * weightPitch[0] + i]);
      }
      tOutput[x * dstPitch[0] + y] = float(sum);
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibMatMulInstThreadedTransposed(void *dstMatrix, void *dstMatrixDims,
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

  uint64_t offsetAIn = coordOut[0]*actPitch[0];
  uint64_t offsetWIn = coordOut[1]*weightPitch[0];

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    float sum = 0;
    for (unsigned int i = 0; i < actIndex[1]; i++) {
      sum += float(tAInput[offsetAIn + i]) * float(tWInput[offsetWIn + i]);
    }
    tOutput[offsetOut] = float(sum);
    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] != 0) {
      offsetWIn += weightPitch[0];
    }
    else {
      offsetWIn = 0;
      offsetAIn += actPitch[0];
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value, std::size_t>::type = 0>
void matmulOpTrans(uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValues[], float *scale, int32_t *offset){

#define MATMUL_ITERATION               \
    "flw.ps   f0, 0x0(%[actAddr])\n"   \
    "flw.ps   f1, 0x0(%[wgtAddr])\n"   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"        // Mask m0 is set so all lanes are active.
    "xor t0, t0, t0\n"                // The int register t0 is set to 0x0: it will count iterations.
    "fxor.pi f31, f31, f31\n"         // Vectorial register f31 set to 0x0. Only useful lanes: e0, e4.

    "1:\n"                            // New loop (tag 1): vectorised scalar product.
    "addi     t0, t0, 8\n"              // t0 += 8.
    "ble      %[elemsRow], t0, 2f\n"    // if (elemsRow <= t0), forward to tag 2.
    MATMUL_ITERATION                    // The scalar product of the act and weights is added to f31.
    "addi %[actAddr], %[actAddr], 0x20\n"
    "addi %[wgtAddr], %[wgtAddr], 0x20\n"
    "beq      zero, zero, 1b\n"       // Go back to tag 1.

    "2:\n"                            // Tag 2: a new mask is set to finish the row's product.
    "fxor.pi  f0, f0, f0\n"           // f0 is set to 0's to get a correct final matmul iteration.
    "addi     t0, t0, -8\n"           // In these two instructions,
    "sub      t0, %[elemsRow], t0\n"  // we update t0 = elemsRow - (t0 - 8).
    "addi     t1, zero, 1\n"          // t1 is set to 1.
    "sll      t1, t1, t0\n"           // Shift Left Logical t0 positions: t1 = 2^(t0).
    "addi     t1, t1, -1\n"           // Finally, t1 = 2^(t0) - 1.
    "mov.m.x  m0, t1, 0\n"            // The mask is set to t1, so the first t0 lanes are active.
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"        // Finally, %[sum] = f31.e0 + f31.e4.
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fsw.ps f31, 0x0(%[dstAddr])\n"

    :
    : [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr)
    : "t0", "t1", "f0", "f1", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, std::size_t>::type = 0>
void matmulOpTrans(uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValues[], float *scale, int32_t *offset){

#define MATMUL_ITERATION               \
    "fgh.ps   f0, f30(%[actAddr])\n"   \
    "fgh.ps   f1, f30(%[wgtAddr])\n"   \
    "fcvt.ps.f16 f0, f0 \n"            \
    "fcvt.ps.f16 f1, f1 \n"            \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"        // Mask m0 is set so all lanes are active.
    "xor t0, t0, t0\n"                // The int register t0 is set to 0x0: it will count iterations.
    "flw.ps f30, 0x0(%[gthValues])\n" // The gatherValues vector is loaded to f30, one int32 per lane.
    "fxor.pi f31, f31, f31\n"         // Vectorial register f31 set to 0x0. Only useful lanes: e0, e4.

    "1:\n"                            // New loop (tag 1): vectorised scalar product.
    "addi     t0, t0, 8\n"              // t0 += 8.
    "ble      %[elemsRow], t0, 2f\n"    // if (elemsRow <= t0), forward to tag 2.
    MATMUL_ITERATION                    // The scalar product of the act and weights is added to f31.
    "faddi.pi f30, f30, 0x10\n"         // The gather offset values are updated adding 8 positions.
    "beq      zero, zero, 1b\n"       // Go back to tag 1.

    "2:\n"                            // Tag 2: a new mask is set to finish the row's product.
    "fxor.pi  f0, f0, f0\n"           // f0 is set to 0's to get a correct final matmul iteration.
    "addi     t0, t0, -8\n"           // In these two instructions,
    "sub      t0, %[elemsRow], t0\n"  // we update t0 = elemsRow - (t0 - 8).
    "addi     t1, zero, 1\n"          // t1 is set to 1.
    "sll      t1, t1, t0\n"           // Shift Left Logical t0 positions: t1 = 2^(t0).
    "addi     t1, t1, -1\n"           // Finally, t1 = 2^(t0) - 1.
    "mov.m.x  m0, t1, 0\n"            // The mask is set to t1, so the first t0 lanes are active.
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"        // Finally, %[sum] = f31.e0 + f31.e4.
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "fcvt.f16.ps f31, f31\n"           // Conversion fp32 >> fp16.
    "mov.m.x m0, zero, 0x1\n"
    "fsc32h.ps f31, zero(%[dstAddr])\n"

    :
    : [gthValues] "r" (gatherValues),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr)
    : "t0", "t1", "f0", "f1", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value, std::size_t>::type = 0>
void matmulOpTrans(uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValues[], float *scale, int32_t *offset){

#define INT8_TO_FP32(_reg)                  \
    "fsub.pi " #_reg ", " #_reg ", f28 \n"  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"    \
    "fmul.ps " #_reg ", " #_reg ", f29 \n"

#define MATMUL_ITERATION             \
    "fgb.ps   f0, f30(%[actAddr])\n" \
    "fgb.ps   f1, f30(%[wgtAddr])\n" \
    "fbc.ps f28, 0x0(%[offset]) \n"  \
    "fbc.ps f29, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                 \
    "fbc.ps f28, 0x4(%[offset]) \n"  \
    "fbc.ps f29, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                 \
    "fmul.ps    f0, f0, f1\n"        \
    "fswizz.ps  f1, f0, 0xe\n"       \
    "fadd.ps    f0, f0, f1\n"        \
    "fswizz.ps  f1, f0, 0x1\n"       \
    "fadd.ps    f0, f0, f1\n"        \
    "fadd.ps    f31, f0, f31\n"

#define FP32_TO_INT8(_reg)                      \
    "frcp.ps f29, f29 \n"                       \
    "fcvt.ps.pw f28, f28 \n"                    \
    "fmadd.ps " #_reg ", " #_reg ", f29, f28 \n" \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsat8.pi " #_reg ", " #_reg "\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"        // Mask m0 is set so all lanes are active.
    "xor t0, t0, t0\n"                // The int register t0 is set to 0x0: it will count iterations.
    "flw.ps f30, 0x0(%[gthValues])\n" // The gatherValues vector is loaded to f30, one int32 per lane.
    "fxor.pi f31, f31, f31\n"         // Vectorial register f31 set to 0x0. Only useful lanes: e0, e4.

    "1:\n"                            // New loop (tag 1): vectorised scalar product.
    "addi     t0, t0, 8\n"              // t0 += 8.
    "ble      %[elemsRow], t0, 2f\n"    // if (elemsRow <= t0), forward to tag 2.
    MATMUL_ITERATION                    // The scalar product of the act and weights is added to f31.
    "faddi.pi f30, f30, 0x8\n"          // The gather offset values are updated adding 8 positions.
    "beq      zero, zero, 1b\n"       // Go back to tag 1.

    "2:\n"                            // Tag 2: a new mask is set to finish the row's product.
    "fxor.pi  f0, f0, f0\n"           // f0 is set to 0's to get a correct final matmul iteration.
    "addi     t0, t0, -8\n"           // In these two instructions,
    "sub      t0, %[elemsRow], t0\n"  // we update t0 = elemsRow - (t0 - 8).
    "addi     t1, zero, 1\n"          // t1 is set to 1.
    "sll      t1, t1, t0\n"           // Shift Left Logical t0 positions: t1 = 2^(t0).
    "addi     t1, t1, -1\n"           // Finally, t1 = 2^(t0) - 1.
    "mov.m.x  m0, t1, 0\n"            // The mask is set to t1, so the first t0 lanes are active.
    MATMUL_ITERATION

    "fmvs.x.ps t0, f31, 0x4\n"        // Finally, sum = f31.e0 + f31.e4.
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"       // Now, the sum is in f31.e0.
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f28, 0x8(%[offset]) \n"
    "fbc.ps f29, 0x8(%[scale]) \n"
    FP32_TO_INT8(f31)
    "fsc32b.ps f31, zero(%[dstAddr])\n"

    :
    : [gthValues] "r" (gatherValues),
      [elemsRow] "r" (elemsRow),
      [offset] "r" (offset),
      [scale] "r" (scale),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr)
    : "t0", "t1", "f0", "f1", "f28", "f29", "f30", "f31", "memory");

#undef INT8_TO_FP32
#undef MATMUL_ITERATION
#undef FP32_TO_INT8

}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value && !std::is_same<srcType, float16>::value && !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void matmulOpTrans (uintptr_t dstAddr, intptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValues[], float *scale, int32_t *offset){}

// Version assuming the weights tensor is transposed. Used for CONSTANT tensors
template <typename srcType>
void dnn_lib::fwdLibMatMulInstVectorizedTransposed(void *dstMatrix, void *dstMatrixDims,
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

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int last_non_zero_coord;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, last_non_zero_coord);

  uint64_t offsetOut = 0;
  for (unsigned int i = 0; i < last_non_zero_coord; i++) {
    offsetOut += dstPitch[i]*coordOut[i];
  }

  uint64_t offsetAIn = coordOut[0]*actPitch[0];
  uint64_t offsetWIn = coordOut[1]*weightPitch[0];

  int32_t gatherValues[8];
  if (typeSize < 4) {
    gatherValues[0] = 0;
    for (unsigned int i = 1; i < 8; ++i)
      gatherValues[i] = gatherValues[i - 1] + typeSize;
  }

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    uintptr_t dstAddr = (uintptr_t)dstMatrix + typeSize*offsetOut;
    uintptr_t actAddr = (uintptr_t)activations + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)weights + typeSize*offsetWIn;
    matmulOpTrans <srcType>(dstAddr, actAddr, wgtAddr, actIndex[1], gatherValues, scale, offset);
    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] != 0) {
      offsetWIn += weightPitch[0];
    }
    else {
      offsetWIn = 0;
      offsetAIn += actPitch[0];
    }
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

GEN_INSTANCES_OP(template, fwdLibMatMulInstTransposed, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibMatMulInstThreadedTransposed, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibMatMulInstVectorizedTransposed, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         float *scale, int32_t *offset, uint64_t flags);
