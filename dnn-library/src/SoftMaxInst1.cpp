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

  // Hypothesis 1 (SHAPE): both source and destination tensors have the same
  // padding pattern (and the same pitches, since they have the same
  // dimensions). Hypothesis 2 (COHERENCE): each row of the source tensor
  // contains an integer number of cl's, that is, srcPitch[0] is a multiple of
  // the cache line length. Thus, dividing the minions' work by rows guarantees
  // there will not be two minions in different rows writing on the same cache
  // line.

template <typename srcType>
void dnn_lib::fwdLibSoftMaxInstThreaded1(void *dstT, void *srcT, void *srcTDims,
                                         void *srcTPitches, const float *scale,
                                         const int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  const Addresser<srcType> acumInt(dstT, scale[1], offset[1]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  size_t typeSize = getsize<srcType>();

  unsigned int *srcIndex = (unsigned int *)srcTDims;
  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  float e, sum, inverseSum;

  unsigned int rowstodo = srcIndex[0] / activeMinions;
  unsigned int firstrow = minionId * rowstodo;
  unsigned int lastrow = firstrow + rowstodo;

  for (unsigned int n = firstrow; n < lastrow; n++) {
    unsigned int start = n * srcPitch[0];
    unsigned int end = start + srcIndex[1];

    float max = float(tInput[start]);
    for (unsigned int i = start + 1; i < end; i++)
      max = std::max(max, float(tInput[i]));

    // Compute exp.
    sum = 0;
    for (unsigned int i = start; i < end; i++) {
      e = getExp(float(tInput[i]) - max);
      sum += e;
      tOutput[i] = float(e); // here, the shape hypothesis is important.
    }

    fpReciprocalSingleElement(sum, inverseSum);
    // Normalize the output.
    for (unsigned int i = start; i < end; i++) {
      auto in = acumInt[i];
      in = in * inverseSum;
      tOutput[i] = in;
    }
  }

  unsigned int cll = 64 / getsize<srcType>();
  unsigned int clperpitch = srcPitch[0] / cll;
  if (DO_EVICTS) {
    unsigned int clperminion = clperpitch * rowstodo;
    unsigned int initialAddr = firstrow * srcPitch[0];
    evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
  }

  unsigned int doneRows = activeMinions * rowstodo;
  unsigned int remainingrows = srcIndex[0] - doneRows;

  if (remainingrows != 0) {
    //    unsigned int cll = 64/sizeof(srcType);
    unsigned int clperrow = (srcIndex[1] - 1) / cll + 1;

    unsigned int minionsperrow = 1;
    int level = -1; // level = log2(minionsperrow) - 1. This is a useful
                    // parameter for reducing and broadcasting.
    unsigned int aux =
        activeMinions /
        (2 * remainingrows); // This parameter helps guarantee that there are
                             // enough minions to double minionsperrow.
    while ((minionsperrow <= aux) && (minionsperrow <
               clperrow)) { // If possible (mpr < aux), we double mpr until we
                           // have at least 1 minion per cl.
      minionsperrow *= 2;
      ++level;
    }

    if (minionId >= minionsperrow * remainingrows)
      return;

    unsigned int moreRows = minionId / minionsperrow;
    unsigned int minioninrow = minionId - moreRows*minionsperrow;
    unsigned int type1minions = clperrow % minionsperrow;
    unsigned int clperminion;
    unsigned int K; // Number of skipped tensor elements by a minion in its own row.
    if (minioninrow < type1minions) {
      clperminion = ((clperrow - 1) / minionsperrow) + 1;
      K = minioninrow * clperminion * cll;
    } else {
      clperminion = clperrow / minionsperrow;
      K = (type1minions + minioninrow * clperminion) * cll;
    }

    // Starting and ending positions in the tensor which the minion will write on.
    unsigned int start = (doneRows + moreRows) * srcPitch[0] + K;
    unsigned int end = start + clperminion * cll;
    // If the minion is the last working minion in its row, its ending position
    // should be modified so it avoids padding.
    if ((clperrow < minionsperrow) && (minioninrow == clperrow - 1) ||
        (clperrow >= minionsperrow) && (minioninrow == minionsperrow - 1))
      end = start - K + srcIndex[1];

    // Now, we perform the SoftMax operation, using shared information between
    // minions.
    float max = float(
        tInput[start - K]); // Obseration: this way, if a minion has start =
                            // end (it is not assigned any cl's), the
                            // initialization of max will not affect the
                            // maximum value of its row (when reducing).
    for (unsigned int i = start; i < end; ++i)
      max = std::max(max, float(tInput[i]));

    // After reducing and broadcasting, the variable max will be the maximum
    // value in each the minion's row.
    for (int i = 0; i <= level; i++) {
      max = tensor_reduce_float(max, 0x2, 1, i, 0x3);
    }
    for (int i = level; i >= 0; i--) {
      max = tensor_reduce_float(max, 0x8, 1, i, 0x2);
    }

    sum = 0;
    for (unsigned int i = start; i < end; i++) {
      e = getExp(float(tInput[i]) - max);
      sum += e;
      tOutput[i] = float(e);
    }
    // Again, after reducing and broadcasting, the variable sum will be the
    // total sum of each the minion's row.
    for (int i = 0; i <= level; i++) {
      sum = tensor_reduce_float(sum, 0x0, 1, i, 0x3);
    }
    if (minionId % minionsperrow == 0)
      fpReciprocalSingleElement(
          sum, inverseSum); // only the first minion must do this calculation,
                            // for saving power and stores.
    for (int i = level; i >= 0; i--) {
      inverseSum = tensor_reduce_float(inverseSum, 0x8, 1, i, 0x2);
    }

    // Finally, the output is normalized.
    for (unsigned int i = start; i < end; i++) {
      auto in = acumInt[i];
      in = in * inverseSum;
      tOutput[i] = in;
    }
    if (clperminion > 0 && DO_EVICTS)
      evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*start, clperminion);
  }
}

// Vectorized version: same hypothesis than SoftMaxInstThreaded1.
// Possible source types: fp16, fp32 (and output of the same type).
// TODO: use templates for each srcType for a speed up (the only
// difference is in the GATHER_FLOAT and SCATTER_FLOAT functions).

template <typename srcType>
void dnn_lib::fwdLibSoftMaxInstVectorized1(void *dstT, void *srcT, void *srcTDims,
                                           void *srcTPitches, const float *scale,
                                           const int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  const Addresser<srcType> acumInt(dstT, scale[1], offset[1]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);

  unsigned int *srcIndex = (unsigned int *)srcTDims;
  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  size_t typeSize = getsize<srcType>();
  float e, sum, inverseSum;

  unsigned int rowstodo = srcIndex[0] / activeMinions;
  unsigned int firstrow = minionId * rowstodo;
  unsigned int lastrow = firstrow + rowstodo;

  unsigned int step = srcPitch[0]*typeSize;
  unsigned int memOffset = firstrow*step;
  uintptr_t srcAddr = (uintptr_t)srcT + memOffset;
  uintptr_t dstAddr = (uintptr_t)dstT + memOffset;

  unsigned int numRegs = srcIndex[1]/8;
  unsigned int extraLanes = srcIndex[1] - 8*numRegs;
  bool floatType = (typeSize == 4); // 1 if fp32 and 0 if fp16.
  int32_t registerSize = 8*typeSize;
  float log2e = M_LOG2E;

#define GATHER_FLOAT(_addr)                       \
      "beq %[floatType], zero, 16f \n"            \
      "flw.ps f0, 0x0(" #_addr ") \n"             \
      "j 32f \n"                                  \
      "16: \n"                                    \
      "fg32h.ps f0, t0(" #_addr ") \n"            \
      "fcvt.ps.f16 f0, f0 \n"                     \
      "32: \n"

#define SCATTER_FLOAT                             \
      "beq %[floatType], zero, 16f \n"            \
      "fsw.ps f0, 0x0(%[dstAddr]) \n"             \
      "j 32f \n"                                  \
      "16: \n"                                    \
      "fcvt.f16.ps f0, f0 \n"                     \
      "fsc32h.ps f0, t0(%[dstAddr]) \n"           \
      "32: \n"

#define EXP                                       \
      "fadd.ps f0, f0, f29 \n"                    \
      "fmul.ps f0, f0, f17 \n"                    \
      "fexp.ps f0, f0 \n"

#define DO_REG(_op, _reg)                         \
      "mov.m.x m0, zero, 0xff \n"                 \
      "fswizz.ps    f17, " #_reg ", 0xe \n"       \
      "f" #_op ".ps " #_reg ", f17, " #_reg " \n" \
      "fswizz.ps    f17, " #_reg ", 0x1 \n"       \
      "f" #_op ".ps " #_reg ", f17, " #_reg " \n" \
      "fmvs.x.ps    t1, " #_reg ", 0x4 \n"        \
      "fmv.w.x      f17, t1 \n"                   \
      "f" #_op ".s  " #_reg ", f17, " #_reg " \n" \
      "fmvs.x.ps    t1, " #_reg ", 0x0 \n"        \
      "fbcx.ps      " #_reg ", t1 \n"

  for (unsigned int n = firstrow; n < lastrow; n++) {

    __asm__ __volatile__(
      "mov.m.x m0, zero, 0xff \n"
      "fxor.pi f28, f28, f28 \n"
      SET_MINUS_INFTY(f29)

///// PART 1: COMPUTATION OF THE MAX VALUE IN ROW.
      "addi t1, zero, 0x0 \n"
      "add t2, zero, %[srcAddr] \n"
      "add t3, zero, %[dstAddr] \n"

      "addi t0, zero, 0x1 \n"
      "beq %[floatType], t0, 1f \n"
      "li t0, %[gs32_offsets]\n"

      "1: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 2f \n"
      GATHER_FLOAT(%[srcAddr])
      "fmax.ps f29, f0, f29 \n"
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 1b \n"

      "2: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 3f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[srcAddr])
      "fmax.ps f29, f0, f29 \n"

      "3: \n" // Computation of the max value.
      DO_REG(max, f29)
      "fsub.ps f29, f28, f29 \n" // f29 = -max.

///// PART 2: COMPUTATION OF EXPONENTIALS AND ITS SUM.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddr], zero, t2 \n"
      "add %[dstAddr], zero, t3 \n"

      "fbc.ps f17, 0x0(%[log2e]) \n"

      "4: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 5f \n"
      GATHER_FLOAT(%[srcAddr])
      EXP
      "fadd.ps f28, f0, f28 \n"
      SCATTER_FLOAT
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 4b \n"

      "5: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 6f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[srcAddr])
      EXP
      "fadd.ps f28, f0, f28 \n"
      SCATTER_FLOAT

      "6: \n" // Computation of the sum of exponentials.
      DO_REG(add, f28)
      "frcp.ps f28, f28 \n" // Reciprocal of the sum of exps.

///// PART 3: PRODUCT OF THE EXPONENTIALS BY THE INVERSE SUM.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddr], zero, t2 \n"
      "add %[dstAddr], zero, t3 \n"

      "7: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 8f \n"
      GATHER_FLOAT(%[dstAddr])
      "fmul.ps f0, f0, f28 \n"
      SCATTER_FLOAT
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 7b \n"

      "8: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 9f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[dstAddr])
      "fmul.ps f0, f0, f28 \n"
      SCATTER_FLOAT

      "9: \n"

      :
      : [log2e] "r" (&log2e),
        [registerSize] "r" (registerSize),
        [floatType] "r" (floatType),
        [srcAddr] "r" (srcAddr),
        [dstAddr] "r" (dstAddr),
        [numRegs] "r" (numRegs),
        [extraLanes] "r" (extraLanes),
        [gs32_offsets] "i" (  fg32b_conf )
      : "t0", "t1", "t2", "t3", "f0", "f17", "f28", "f29", "memory");

    srcAddr += step;
    dstAddr += step;
  }

  unsigned int doneRows = activeMinions * rowstodo;
  unsigned int remainingrows = srcIndex[0] - doneRows;

  if (remainingrows != 0) {

    unsigned int cll = 64 / getsize<srcType>();
    unsigned int clperrow = (srcIndex[1] - 1) / cll + 1;

    unsigned int minionsperrow = 1;
    int level = -1;
    unsigned int aux = activeMinions / (2 * remainingrows);

 // If possible (mpr < aux), we double mpr until we have at least 1 minion per cl.
    while ((minionsperrow <= aux) && (minionsperrow < clperrow)) {
      minionsperrow *= 2;
      ++level;
    }

    if (minionId >= minionsperrow * remainingrows)
      return;

    unsigned int moreRows = minionId / minionsperrow;
    unsigned int minioninrow = minionId - moreRows*minionsperrow;
    unsigned int type1minions = clperrow % minionsperrow;
    unsigned int clperminion;
    unsigned int K; // Number of skipped tensor elements by a minion in its own row.
    if (minioninrow < type1minions) {
      clperminion = ((clperrow - 1) / minionsperrow) + 1;
      K = minioninrow * clperminion * cll;
    } else {
      clperminion = clperrow / minionsperrow;
      K = (type1minions + minioninrow * clperminion) * cll;
    }

    // Starting and ending positions in the tensor which the minion will write on.
    unsigned int start = (doneRows + moreRows) * srcPitch[0] + K;
    unsigned int end = start + clperminion * cll;
    // If the minion is the last working minion in its row, its ending position
    // should be modified so it avoids padding.
    if ((clperrow < minionsperrow) && (minioninrow == clperrow - 1) ||
        (clperrow >= minionsperrow) && (minioninrow == minionsperrow - 1))
      end = start - K + srcIndex[1];

    memOffset = start*typeSize;
    srcAddr = (uintptr_t)srcT + memOffset;
    dstAddr = (uintptr_t)dstT + memOffset;
    numRegs = (end - start)/8;
    extraLanes = (end - start) - 8*numRegs;

    uint64_t csr_enc = ((0ULL  & 0x2) << 62)        |
                       ((29ULL & 0x1F) << 57)       |   // Register: f29.
                       ((0ULL  & 0x1FFFFFFF) << 28) |
                       ((2ULL  & 0xF) << 24)        |   // Op: 0 = add, 2 = max
                       ((1ULL  & 0xFF) << 16)       |   // Number of registers
                       ((0ULL  & 0x1FFF) << 3)      |   // Tree depth
                       ((0ULL  & 0x1) << 2)         |
                       ((0x3 & 0x3));                   // 2: broadcast, 3:reduce

#define REDUCE                                    \
      "addi t4, zero, 0x0 \n"                     \
      "1: \n"                                     \
      "blt %[level], t4, 2f \n"                   \
      "csrw tensor_reduce, t1 \n"                 \
      "addi t1, t1, 0x8 \n"                       \
      "addi t4, t4, 0x1 \n"                       \
      "j 1b \n"                                   \
      "2: \n"                                     \
      "addi t1, t1, -8 \n"                        \
      "addi t4, t4, -1 \n"

#define BROADCAST                                 \
      "1: \n"                                     \
      "blt t4, zero, 2f \n"                       \
      "csrw tensor_reduce, t1 \n"                 \
      "addi t1, t1, -8 \n"                        \
      "addi t4, t4, -1 \n"                        \
      "j 1b \n"                                   \
      "2: \n"

    __asm__ __volatile__(
      "mov.m.x m0, zero, 0xff \n"
      "fxor.pi f28, f28, f28 \n"
      SET_MINUS_INFTY(f29)

///// PART 1: COMPUTATION OF THE MAX VALUE IN ROW.
      "addi t1, zero, 0x0 \n"
      "add t2, zero, %[srcAddr] \n"
      "add t3, zero, %[dstAddr] \n"

      "addi t0, zero, 0x1 \n"
      "beq %[floatType], t0, 1f \n"
      "li t0, %[g32_conf]\n"

      "1: \n" // Coverage of the full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 2f \n"
      GATHER_FLOAT(%[srcAddr])
      "fmax.ps f29, f0, f29 \n"
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 1b \n"

      "2: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 3f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[srcAddr])
      "fmax.ps f29, f0, f29 \n"

      "3: \n" // Computation of the max value.
      DO_REG(max, f29)

      "ld t1, 0x0(%[csr_enc]) \n"
      REDUCE
      // Changing reduce max to broadcast get.
      "addi t1, t1, -1 \n"
      "addi t5, zero, 6 \n"
      "slli t5, t5, 0x18 \n"
      "add t1, t1, t5 \n"
      BROADCAST

      "fsub.ps f29, f28, f29 \n" // f29 = -max.

///// PART 2: COMPUTATION OF EXPONENTIALS AND ITS SUM.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddr], zero, t2 \n"
      "add %[dstAddr], zero, t3 \n"

      "fbc.ps f17, 0x0(%[log2e]) \n"

      "4: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 5f \n"
      GATHER_FLOAT(%[srcAddr])
      EXP
      "fadd.ps f28, f0, f28 \n"
      SCATTER_FLOAT
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 4b \n"

      "5: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 6f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[srcAddr])
      EXP
      "fadd.ps f28, f0, f28 \n"
      SCATTER_FLOAT

      "6: \n" // Computation of the sum of exponentials.
      DO_REG(add, f28)

      "ld t1, 0x0(%[csr_enc]) \n"
      // Changing f29 to f28.
      "addi t4, zero, 1 \n"
      "slli t4, t4, 0x39 \n"
      "sub t1, t1, t4 \n"
      // Changing max to add.
      "addi t4, zero, 2 \n"
      "slli t4, t4, 0x18 \n"
      "sub t1, t1, t4 \n"
      REDUCE
      // Changing reduce add to broadcast get.
      "addi t1, t1, -1 \n"
      "addi t5, zero, 8 \n"
      "slli t5, t5, 0x18 \n"
      "add t1, t1, t5 \n"
      BROADCAST

      "frcp.ps f28, f28 \n" // Reciprocal of the sum of exps.

///// PART 3: PRODUCT OF THE EXPONENTIALS BY THE INVERSE SUM.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddr], zero, t2 \n"
      "add %[dstAddr], zero, t3 \n"

      "7: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 8f \n"
      GATHER_FLOAT(%[dstAddr])
      "fmul.ps f0, f0, f28 \n"
      SCATTER_FLOAT
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 7b \n"

      "8: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 9f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[dstAddr])
      "fmul.ps f0, f0, f28 \n"
      SCATTER_FLOAT

      "9: \n"

      :
      : [log2e] "r" (&log2e),
        [registerSize] "r" (registerSize),
        [floatType] "r" (floatType),
        [srcAddr] "r" (srcAddr),
        [dstAddr] "r" (dstAddr),
        [numRegs] "r" (numRegs),
        [extraLanes] "r" (extraLanes),
        [csr_enc] "r" (&csr_enc),
        [level] "r" (level),
        [g32_conf] "i" (fg32h_conf)
      : "t0", "t1", "t2", "t3", "t4", "t5", "f0", "f17", "f28", "f29", "memory");

  }
#undef GATHER_FLOAT
#undef SCATTER_FLOAT
#undef EXP
#undef DO_REG
#undef REDUCE
#undef BROADCAST
}

GEN_INSTANCES_OP(template, fwdLibSoftMaxInstThreaded1, void *dstT, void *srcT, void *srcTDims,
                          void *srcTPitches, const float *scale, const int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibSoftMaxInstVectorized1, void *dstT, void *srcT, void *srcTDims,
                          void *srcTPitches, const float *scale, const int32_t *offset, uint64_t flags);
