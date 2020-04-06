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

#ifndef _ROWWISE_QUANTIZED_FULLY_CONNECTED_H_
#define _ROWWISE_QUANTIZED_FULLY_CONNECTED_H_

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

inline void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTy(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  int8_t *tWInput = (int8_t *)pweights;
  int32_t *tBias = (int32_t *)pbias;
  float *tScale = (float *)pscale;
  int32_t *tOffset = (int32_t *)poffset;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  float inversedDstScale;
  fpReciprocalSingleElement(dstscale, inversedDstScale);
  for (size_t i = 0; i < dataIndex[0]; i++) {
    for (size_t j = 0; j < dstIndex[1]; j++) {
      float matMulScale = tScale[j] * srcscale;
      float inversedMatMulScale;
      fpReciprocalSingleElement(matMulScale, inversedMatMulScale);
      int32_t sum = 0;
      for (size_t k = 0; k < dataIndex[1]; k++) {
        int32_t W = tWInput[j * weightPitch[0] + k];
        int32_t A = tAInput[i * dataPitch[0] + k];
        sum += (W - tOffset[j]) * (A - srcoffset);
      }
      int32_t B = nearbyintf(float(tBias[j] - biasoffset) *
                             (biasscale * inversedMatMulScale));
      sum += B;
      // Scale the result back to the expected destination scale.
      tOutput[i * dstPitch[0] + j] = clip<int32_t, int8_t>(nearbyintf(
          float(sum) * (matMulScale * inversedDstScale) + dstoffset));
    }
  }
}


inline void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  int8_t *tWInput = (int8_t *)pweights;
  int32_t *tBias = (int32_t *)pbias;
  float *tScale = (float *)pscale;
  int32_t *tOffset = (int32_t *)poffset;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  float invDstScale;
  fpReciprocalSingleElement(dstscale, invDstScale);

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<int8_t>();
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

  uint64_t offsetAIn = coordOut[0]*dataPitch[0];
  uint64_t offsetWIn = coordOut[1]*weightPitch[0];

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  while (!done && (offsetOut < posMax)) {

    float matMulScale = tScale[coordOut[1]] * srcscale;
    float invMatMulScale;
    fpReciprocalSingleElement(matMulScale, invMatMulScale);
    int32_t sum = nearbyintf(float(tBias[coordOut[1]] - biasoffset) * biasscale * invMatMulScale);
    int32_t woffset = tOffset[coordOut[1]];
    for (size_t k = 0; k < dataIndex[1]; k++) {
      int32_t W = tWInput[offsetWIn + k];
      int32_t A = tAInput[offsetAIn + k];
      sum += (W - woffset) * (A - srcoffset);
    }
    tOutput[offsetOut] = clip<int32_t, int8_t>(nearbyintf(
      float(sum) * (matMulScale * invDstScale) + dstoffset));

    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] != 0) {
      offsetWIn += weightPitch[0];
    }
    else {
      offsetWIn = 0;
      offsetAIn += dataPitch[0];
    }

  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);
}

inline void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyVectorized(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  int8_t *tWInput = (int8_t *)pweights;
  int32_t *tBias = (int32_t *)pbias;
  float *tScale = (float *)pscale;
  int32_t *tOffset = (int32_t *)poffset;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  float invDstScale;
  fpReciprocalSingleElement(dstscale, invDstScale);

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<int8_t>();
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

  uint64_t offsetAIn = coordOut[0]*dataPitch[0];
  uint64_t offsetWIn = coordOut[1]*weightPitch[0];

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  while (!done && (offsetOut < posMax)) {

    float matMulScale = tScale[coordOut[1]] * srcscale;
    float invMatMulScale;
    fpReciprocalSingleElement(matMulScale, invMatMulScale);
    int32_t sum = nearbyintf(float(tBias[coordOut[1]] - biasoffset) * biasscale * invMatMulScale);
    int32_t woffset = tOffset[coordOut[1]];

    uintptr_t actAddr = (uintptr_t)tAInput + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)tWInput + typeSize*offsetWIn;

#define MATMUL_ITERATION               \
    "fgb.ps   f0, f28(%[actAddr])\n" \
    "fgb.ps   f1, f28(%[wgtAddr])\n" \
    "fsub.pi    f0, f0, f29\n"         \
    "fsub.pi    f1, f1, f30\n"         \
    "fmul.pi    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.pi    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.pi    f0, f0, f1\n"          \
    "fadd.pi    f31, f0, f31\n"

    int32_t gatherValues[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ __volatile__(
      "mov.m.x m0, zero, 0xff\n"        // Mask m0 is set so all lanes are
                                        //active.
      "addi t0, %[sum], 0x0\n"          // The value of sum is stored in
                                        //the integer register t0.
      "xor t1, t1, t1\n"                // The int register t1 is set to
                                        //0x0: it will count iterations.
      "flw.ps f28, %[gthValues]\n" // The gatherValues vector is loaded
                                        //to f28, one int32 per lane.
      "fbc.ps f29, 0x0(%[srcoffset])\n" // The int32 srcoffset is broadcast
                                        //to the 8 lanes of f29.
      "fbc.ps f30, 0x0(%[woffset])\n"   // The int32 woffset is broadcast to
                                        //the 8 lanes of f30.
      "fxor.pi f31, f31, f31\n"         // Vectorial register f31 set to 0x0.
                                        //Only useful lanes: e0, e4.

      "1:\n"                            // New loop (tag 1): vectorised scalar
                                        //product.
      "addi     t1, t1, 8\n"            // t1 += 8.
      "ble      %[elemsRow], t1, 2f\n"  // if (elemsRow <= t1), forward to
                                        //tag 2.
      MATMUL_ITERATION                  // The scalar product of the data and
                                        //weights is added to f31.
      "faddi.pi f28, f28, 0x8\n"        // The gather offset values are updated
                                        //adding 8 positions.
      "beq      zero, zero, 1b\n"       // Go back to tag 1.

      "2:\n"                            // Tag 2: a new mask is set to finish
                                        //the row's product.
      "fxor.pi  f0, f0, f0\n"           // f0 is set to 0's to get a correct
                                        //final matmul iteration.
      "addi     t1, t1, -8\n"           // In these two instructions,
      "sub      t1, %[elemsRow], t1\n"  // we update t1 = elemsRow - (t1 - 8).
      "addi     t2, zero, 1\n"          // t2 is set to 1.
      "sll      t2, t2, t1\n"           // Shift Left Logical t1 positions:
                                        //t2 = 2^(t1).
      "addi     t2, t2, -1\n"           // Finally, t2 = 2^(t1) - 1.
      "mov.m.x  m0, t2, 0\n"            // The mask is set to t2, so the first
                                        //t1 lanes are active.
      MATMUL_ITERATION
      "fmvs.x.ps t1, f31, 0x0\n"        // The sum stored in f31.e0 is stored
                                        //in the int register t1.
      "add       t0, t0, t1\n"          // This way, it can be summed to its
                                        //initial value in t0.
      "fmvs.x.ps t1, f31, 0x4\n"        // The same is done for the sum stored
                                        //in f31.e4,
      "add       t0, t0,  t1\n"         // so t0 has now the total value of the
                                        //scalar product.
      "addi      %[sum], t0, 0x0\n"     // The value in t0 is then stored in
                                        //the variable "sum".

      : [sum] "+&r" (sum)
      : [gthValues] "m" (* (const int32_t(*)[8]) gatherValues),
        [srcoffset] "r" (&srcoffset),
        [woffset]   "r" (&woffset),
        [actAddr] "r" (actAddr),
        [wgtAddr] "r" (wgtAddr),
        [elemsRow]  "r" (dataIndex[1])
      : "t0", "t1", "t2", "f0", "f1", "f29", "f30", "f31");

    tOutput[offsetOut] = clip<int32_t, int8_t>(nearbyintf(
      float(sum) * (matMulScale * invDstScale) + dstoffset));

    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] != 0) {
      offsetWIn += weightPitch[0];
    }
    else {
      offsetWIn = 0;
      offsetAIn += dataPitch[0];
    }

  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);

#undef MATMUL_ITERATION
}

inline void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyAligned32Bytes(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  int8_t *tWInput = (int8_t *)pweights;
  int32_t *tBias = (int32_t *)pbias;
  float *tScale = (float *)pscale;
  int32_t *tOffset = (int32_t *)poffset;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  float invDstScale;
  fpReciprocalSingleElement(dstscale, invDstScale);

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<int8_t>();
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

  uint64_t offsetAIn = coordOut[0]*dataPitch[0];
  uint64_t offsetWIn = coordOut[1]*weightPitch[0];

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  while (!done && (offsetOut < posMax)) {

    float matMulScale = tScale[coordOut[1]] * srcscale;
    float invMatMulScale;
    fpReciprocalSingleElement(matMulScale, invMatMulScale);
    int32_t sum = nearbyintf(float(tBias[coordOut[1]] - biasoffset) * biasscale * invMatMulScale);
    int32_t woffset = tOffset[coordOut[1]];

    uintptr_t actAddr = (uintptr_t)tAInput + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)tWInput + typeSize*offsetWIn;

#define MATMUL_ITERATION               \
    "fg32b.ps   f0, t0(%[actAddr])\n"  \
    "fg32b.ps   f1, t0(%[wgtAddr])\n"  \
    "fsub.pi    f0, f0, f29\n"         \
    "fsub.pi    f1, f1, f30\n"         \
    "fmul.pi    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.pi    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.pi    f0, f0, f1\n"          \
    "fadd.pi    f31, f0, f31\n"

    __asm__ __volatile__(
      "mov.m.x m0, zero, 0xff\n"
      "li t0, %[g32_conf]\n"
      "xor t1, t1, t1\n"
      "fbc.ps f29, 0x0(%[srcoffset])\n"
      "fbc.ps f30, 0x0(%[woffset])\n"
      "fxor.pi f31, f31, f31\n"
      "1:\n"
      "addi     t1, t1, 8\n"
      "ble      %[elemsRow], t1, 2f\n"
      MATMUL_ITERATION
      "addi     %[actAddr], %[actAddr], 0x8\n"
      "addi     %[wgtAddr], %[wgtAddr], 0x8\n"
      "j 1b\n"
      "2:\n"
      "fxor.pi  f0, f0, f0\n"
      "addi     t1, t1, -8\n"
      "sub      t1, %[elemsRow], t1\n"
      "addi     t2, zero, 1\n"
      "sll      t2, t2, t1\n"
      "addi     t2, t2, -1\n"
      "mov.m.x  m0, t2, 0\n"
      MATMUL_ITERATION
      "fmvs.x.ps t1, f31, 0x0\n"
      "addi      t0, zero, 0x0\n"
      "add       t0, t0, t1\n"
      "fmvs.x.ps t1, f31, 0x4\n"
      "add       t0, t0,  t1\n"
      "add       %[sum], t0, %[sum]\n"

      : [sum] "=r" (sum)
      : [actAddr] "r" (actAddr),
        [wgtAddr] "r" (wgtAddr),
        [srcoffset] "r" (&srcoffset),
        [woffset]   "r" (&woffset),
        [elemsRow]  "r" (dataIndex[1]),
        [g32_conf] "i" (fg32b_conf)
      : "t0", "t1", "t2", "f0", "f1", "f29", "f30", "f31");

    tOutput[offsetOut] = clip<int32_t, int8_t>(nearbyintf(
      float(sum) * (matMulScale * invDstScale) + dstoffset));

    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] != 0) {
      offsetWIn += weightPitch[0];
    }
    else {
      offsetWIn = 0;
      offsetAIn += dataPitch[0];
    }
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);

#undef MATMUL_ITERATION
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ROWWISE_QUANTIZED_FULLY_CONNECTED_H_
