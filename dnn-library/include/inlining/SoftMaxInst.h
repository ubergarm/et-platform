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

#ifndef _SOFTMAX_INST_H_
#define _SOFTMAX_INST_H_

#include "Addresser.h"
#include "Float16.h"
#include "LibTensor.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {


// Single-thread version with small optimisations. Useful when the padding
// hypothesis are not met.
template <ElemKind elK>
INLINE_ATTR void fwdLibSoftMaxInst(LibTensor* outT, LibTensor* inT, [[maybe_unused]] uint64_t flags,
                                   const uint32_t minionOffset = 0,
                                   [[maybe_unused]] const uint32_t assignedMinions = 0) {

  static_assert(elK == FloatTy or elK == Float16Ty or elK == BFloat16Ty, "Unsupported elK type.");
  assert(inT->getElementType() == elK and outT->getElementType() == elK);
  assert(inT->ndims() == 2 and outT->ndims() == 2);
  assert(inT->dims().data()[0] == outT->dims().data()[0] and inT->dims().data()[1] == outT->dims().data()[1]);

  using srcType = typename elemKind2elemTy<elK>::type;

  if (get_minion_id() != minionOffset) return;

  auto dstT = outT->getRawDataPointer<srcType>();
  auto srcT = inT->getRawDataPointer<srcType>();

  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  const Addresser<elK> acumInt(dstT, outT->getScale(), outT->getOffset());
  const Addresser<elK> tInput(srcT, inT->getScale(), inT->getOffset());

  const dim_t *srcIndex = inT->dims().data();
  const dim_t *srcPitch = inT->strides().data();
  const dim_t *dstPitch = outT->strides().data();

  for (dim_t n = 0; n < srcIndex[0]; n++) {
    dim_t start = n * srcPitch[0];
    dim_t end = start + srcIndex[1];
    dim_t outStart = n * dstPitch[0];

    float max = float(tInput[start]);
    for (dim_t i = start + 1; i < end; i++)
      max = std::max(max, float(tInput[i]));

    // Compute exp.
    float sum = 0;
    for (dim_t i = start, j = outStart; i < end; i++, j++) {
      float e = getExp(float(tInput[i]) - max);
      sum += e;
      if (elK == BFloat16Ty) {
        tOutput[j] = static_cast<srcType>(e);
      } else {
        tOutput[j] = static_cast<float>(e);
      }
    }

    float inverseSum;
    fpReciprocalSingleElement(sum, inverseSum);

    // Normalize the output.
    for (dim_t i = start, j = outStart; i < end; i++, j++) {
      auto in = acumInt[j];
      if (elK == BFloat16Ty) {
        in = static_cast<srcType>(in * inverseSum);
      } else {
        in = static_cast<float>(in * inverseSum);
      }
      tOutput[j] = in;
    }
  }
}

// Vectorized version: same hypothesis than SoftMaxInstThreaded1.
// Possible source types: fp16, fp32 (and output of the same type).
// TODO: use templates for each srcType for a speed up (the only
// difference is in the GATHER_FLOAT and SCATTER_FLOAT functions).

template <ElemKind elK>
INLINE_ATTR void fwdLibSoftMaxInstVectorized(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                             const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  [[maybe_unused]] const size_t cll = CACHE_LINE_BYTES / outT->getElementSize();
  [[maybe_unused]] const size_t numDims = outT->ndims();
  static_assert(elK == FloatTy or elK == Float16Ty or elK == BFloat16Ty);
  assert(inT->getElementType() == elK and outT->getElementType() == elK);
  assert(inT->ndims() == 2 and outT->ndims() == 2);
  assert(inT->dims().data()[0] == outT->dims().data()[0] and inT->dims().data()[1] == outT->dims().data()[1]);
  assert((uintptr_t)outT->getAddress() % CACHE_LINE_BYTES == 0 and numDims >= 2 and
         outT->strides()[numDims - 2] % cll == 0);

  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  auto dstT = outT->getRawDataPointer<srcType>();
  auto srcT = inT->getRawDataPointer<srcType>();

  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  const Addresser<elK> acumInt(dstT, outT->getScale(), outT->getOffset());
  const Addresser<elK> tInput(srcT, inT->getScale(), inT->getOffset());
  
  const dim_t *srcIndex = inT->dims().data();
  const dim_t *srcPitch = inT->strides().data();
  const dim_t *dstPitch = outT->strides().data();
  constexpr size_t typeSize = getsize<srcType>();

  size_t rowstodo = srcIndex[0] / activeMinions;
  size_t rowsRemainder = srcIndex[0] - rowstodo * activeMinions;
  size_t firstrow;

  if (minionId < rowsRemainder) {
    ++rowstodo;
    firstrow = minionId * rowstodo;
  } else {
    firstrow = (rowstodo + 1) * rowsRemainder + (minionId - rowsRemainder) * rowstodo;
  }

  size_t lastrow = firstrow + rowstodo;
  size_t step = srcPitch[0] * typeSize;
  size_t outStep = dstPitch[0] * typeSize;
  size_t memOffset = firstrow * step;
  size_t outMemOffset = firstrow * outStep;

  uintptr_t srcAddr = (uintptr_t)srcT + memOffset;
  uintptr_t dstAddr = (uintptr_t)dstT + outMemOffset;

  size_t numRegs = srcIndex[1] / 8;
  size_t extraLanes = srcIndex[1] - 8 * numRegs;
  bool floatType = (typeSize == 4); // 1 if fp32 and 0 if fp16.
  int32_t registerSize = 8*typeSize;
  auto log2e = static_cast<float>(M_LOG2E);

#define GATHER_FLOAT(_addr)                                                                                            \
  "beq %[floatType], zero, 16f \n"                                                                                     \
  "flw.ps f0, 0(" #_addr ") \n"                                                                                        \
  "j 32f \n"                                                                                                           \
  "16: \n"                                                                                                             \
  "fgh.ps f0, %[indices](" #_addr ") \n"                                                                               \
  "fcvt.ps.f16 f0, f0 \n"                                                                                              \
  "32: \n"

#define SCATTER_FLOAT                                                                                                  \
  "beq %[floatType], zero, 16f \n"                                                                                     \
  "fsw.ps f0, 0(%[dstAddrTmp]) \n"                                                                                     \
  "j 32f \n"                                                                                                           \
  "16: \n"                                                                                                             \
  "fcvt.f16.ps f0, f0 \n"                                                                                              \
  "fsc32h.ps f0, t0(%[dstAddrTmp]) \n"                                                                                 \
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

  constexpr size_t bytesPerElement = Type::getElementSize(elK);
  float indices;
  static const int32_t values[] = {0 * bytesPerElement, 1 * bytesPerElement, 2 * bytesPerElement, 3 * bytesPerElement,
                                   4 * bytesPerElement, 5 * bytesPerElement, 6 * bytesPerElement, 7 * bytesPerElement};

  __asm__ __volatile__("flq2 %[indices], %[values]\n"
                       : [ indices ] "=f"(indices)
                       : [ values ] "m"(*(const int32_t(*)[8])values));

  for (size_t n = firstrow; n < lastrow; n++) {

    uint64_t srcAddrTmp, dstAddrTmp;

    __asm__ __volatile__(
      /* clang-format off */
      "mov.m.x m0, zero, 0xff \n"
      "fxor.pi f28, f28, f28 \n"
      SET_MINUS_INFTY(f29)

///// PART 1: COMPUTATION OF THE MAX VALUE IN ROW.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddrTmp], zero, %[srcAddrInit]\n"
      "add %[dstAddrTmp], zero, %[dstAddrInit]\n"

      "li t0, %[gs32_offsets]\n"

      "1: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 2f \n"
      GATHER_FLOAT(%[srcAddrTmp])
      "fmax.ps f29, f0, f29 \n"
      "add %[srcAddrTmp], %[srcAddrTmp], %[registerSize] \n"
      "add %[dstAddrTmp], %[dstAddrTmp], %[registerSize] \n"
      "j 1b \n"

      "2: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 3f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[srcAddrTmp])
      "fmax.ps f29, f0, f29 \n"

      "3: \n" // Computation of the max value.
      DO_REG(max, f29)
      "fsub.ps f29, f28, f29 \n" // f29 = -max.

///// PART 2: COMPUTATION OF EXPONENTIALS AND ITS SUM.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddrTmp], zero, %[srcAddrInit]\n"
      "add %[dstAddrTmp], zero, %[dstAddrInit]\n"

      "fbc.ps f17, 0x0(%[log2e]) \n"

      "4: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 5f \n"
      GATHER_FLOAT(%[srcAddrTmp])
      EXP
      "fadd.ps f28, f0, f28 \n"
      SCATTER_FLOAT
      "add %[srcAddrTmp], %[srcAddrTmp], %[registerSize] \n"
      "add %[dstAddrTmp], %[dstAddrTmp], %[registerSize] \n"
      "j 4b \n"

      "5: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 6f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[srcAddrTmp])
      EXP
      "fadd.ps f28, f0, f28 \n"
      SCATTER_FLOAT

      "6: \n" // Computation of the sum of exponentials.
      DO_REG(add, f28)
      "frcp.ps f28, f28 \n" // Reciprocal of the sum of exps.

///// PART 3: PRODUCT OF THE EXPONENTIALS BY THE INVERSE SUM.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddrTmp], zero, %[srcAddrInit]\n"
      "add %[dstAddrTmp], zero, %[dstAddrInit]\n"

      "7: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 8f \n"
      GATHER_FLOAT(%[dstAddrTmp])
      "fmul.ps f0, f0, f28 \n"
      SCATTER_FLOAT
      "add %[srcAddrTmp], %[srcAddrTmp], %[registerSize] \n"
      "add %[dstAddrTmp], %[dstAddrTmp], %[registerSize] \n"
      "j 7b \n"

      "8: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 9f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[dstAddrTmp])
      "fmul.ps f0, f0, f28 \n"
      SCATTER_FLOAT

      "9: \n"
/* clang-format off */
      : [srcAddrTmp] "=&r" (srcAddrTmp), [dstAddrTmp] "=&r" (dstAddrTmp)
      : [log2e] "r" (&log2e),
        [registerSize] "r" (registerSize),
        [floatType] "r" (floatType),
        [srcAddrInit] "r" (srcAddr),
        [dstAddrInit] "r" (dstAddr),
        [numRegs] "r" (numRegs),
        [extraLanes] "r" (extraLanes),
        [gs32_offsets] "i" (fg32h_conf),
        [indices] "f" (indices)
      : "t0", "t1", "f0", "f17", "f28", "f29", "memory");

    srcAddr += step;
    dstAddr += outStep;
  }

#undef GATHER_FLOAT
#undef SCATTER_FLOAT
#undef EXP
#undef DO_REG
#undef REDUCE
#undef BROADCAST

  size_t totalBytes = outStep * rowstodo;
  size_t clperminion = (totalBytes + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (DO_EVICTS and clperminion > 0) {
    evict_va_multi(DO_EVICTS, (uintptr_t)dstT + memOffset, clperminion);
  }
}

  
} // namespace inlining

} // namespace dnn_lib

#endif // _SOFTMAX_INST_H_
