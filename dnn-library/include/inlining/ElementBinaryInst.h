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

#ifndef _ELEMENT_BINARY_INST_H_
#define _ELEMENT_BINARY_INST_H_

#include "Float16.h"
#include "LibTensor.h"
#include "LoadStore.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

 namespace dnn_lib {

 namespace inlining {

 ////////////////////////////////////////////////////////////////////////////////
 // macro to instantiate compute depending on the operation
 ////////////////////////////////////////////////////////////////////////////////

#define EB_COMPUTE(OP_, IS_INDEX_, SKIP_CONVERT_, MATCH_x86)                                                           \
  do {                                                                                                                 \
    if constexpr (std::is_same<opType, OP_>::value) {                                                                  \
      if constexpr (IS_INDEX_) {                                                                                       \
        /*compute*/                                                                                                    \
        __asm__ EB_##OP_##_INDEX_COMPUTE;                                                                              \
      } else {                                                                                                         \
        /*convert operands to float */                                                                                 \
        if constexpr (!(SKIP_CONVERT_)) {                                                                              \
          convert<elK, FloatTy>(in1, in1H, in1, in1H, in1Scale, in1Offset, 0, 0);                                      \
          convert<elK, FloatTy>(in2, in2H, in2, in2H, in2Scale, in2Offset, 0, 0);                                      \
        }                                                                                                              \
        /* compute*/                                                                                                   \
        __asm__ EB_##OP_##_FLOAT_COMPUTE;                                                                              \
        /* convert back */                                                                                             \
        if constexpr (!(SKIP_CONVERT_)) {                                                                              \
          convert<FloatTy, elK, !MATCH_x86>(out, outH, out, outH, 0, 0, outScaleRcp, outOffset);                       \
        }                                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  } while (0)

////////////////////////////////////////////////////////////////////////////////
// asm listing for compute depending on the operator
////////////////////////////////////////////////////////////////////////////////
#define EB_EXPAND_SINGLE_INST(x) (x " %0, %1, %2" : "=f"(out) : "f"(in1), "f"(in2))

#define EB_Add_FLOAT_COMPUTE EB_EXPAND_SINGLE_INST("fadd.ps")
#define EB_Add_INDEX_COMPUTE EB_EXPAND_SINGLE_INST("fadd.pi")

#define EB_Sub_FLOAT_COMPUTE EB_EXPAND_SINGLE_INST("fsub.ps")
#define EB_Sub_INDEX_COMPUTE EB_EXPAND_SINGLE_INST("fsub.pi")

#define EB_Mul_FLOAT_COMPUTE EB_EXPAND_SINGLE_INST("fmul.ps")
#define EB_Mul_INDEX_COMPUTE EB_EXPAND_SINGLE_INST("fmul.pi")

#define EB_Min_FLOAT_COMPUTE EB_EXPAND_SINGLE_INST("fmin.ps")
#define EB_Min_INDEX_COMPUTE EB_EXPAND_SINGLE_INST("fmin.pi")

#define EB_Max_FLOAT_COMPUTE EB_EXPAND_SINGLE_INST("fmax.ps")
#define EB_Max_INDEX_COMPUTE EB_EXPAND_SINGLE_INST("fmax.pi")

#define EB_Div_FLOAT_COMPUTE                                                                                           \
  ("frcp.ps %0, %2\n"     /* out=1/in2 */                                                                              \
   "fmul.ps %0, %0, %1\n" /* out=in1/in2 */                                                                            \
   : "=&f"(out)                                                                                                        \
   : "f"(in1), "f"(in2))

#define EB_Div_INDEX_COMPUTE ("invalid")

/* compute like this: x^y = exp( y*log(x))
   with some special cases into account:
    a) if y == 0 => result is 1
    b) if x == 0 => result i 0
    c) if x < 0 (excluding -inf), NaN if y is not integer.
 */

/* fclass, bit 1 is negative (it does not include -inf)*/
#define NEG_MASK (1 << 1)

#define EB_Pow_FLOAT_COMPUTE                                                                                           \
  ("fmv.w.x %[out], x0\n"                                   /* out=0 (all lanes)*/                                     \
   "maskand m7, m0, m0\n" /* save initial mask for later */ /* m0&=(x!=0) =>  disable lanes where x==0 (case b)*/      \
   "feqm.ps m1, %[x], %[out]\n"                             /*m1= (x==0)*/                                             \
   "feqm.ps m2, %[y], %[out]\n"                             /*m2= (y==0)*/                                             \
   "masknot m2, m2\n"                                       /*m2= (y!=0)*/                                             \
   "maskand m1, m1, m2\n"                                   /* m1 = (x==0, y!=0) */                                    \
   "masknot m1, m1 \n"                                                                                                 \
   "maskand m0, m0, m1\n" /* m0 =  mask with 0's where x==0, y!=0*/                                                    \
                                                                                                                       \
   /* m2 = 1 if y is integer (for case c)*/                                                                            \
   "fround.ps ft0, %[y]\n"                                                                                             \
   "feqm.ps m2, ft0, %[y]\n" /* m3 = 1 if round(y) is odd (for case c) need to exclude inf*/                           \
   "fcvt.pw.ps ft0, ft0\n"                                                                                             \
   "fandi.pi ft0, ft0, 1\n"                                                                                            \
   "fsetm.pi m3, ft0\n" /* exclude inf */                                                                              \
   "fbcx.ps ft2, %[inf]\n"                                                                                             \
   "feqm.ps m6, %[y], ft2\n"                                                                                           \
   "masknot m6, m6 \n"                                                                                                 \
   "maskand m3, m3, m6\n"                                                                                              \
                                                                                                                       \
   /* m4 = x <0 , excluding -inf and denormals*/                                                                       \
   "fclass.ps ft1, %[x]\n"                                                                                             \
   "fbcx.ps ft2, %[neg_m]\n"                                                                                           \
   "fand.pi ft3, ft1, ft2\n"                                                                                           \
   "fsetm.pi m4, ft3\n"                                                                                                \
                                                                                                                       \
   /* m5 = x < 0 (all, used to set sign)  */                                                                           \
   "fltm.ps m5, %[x], %[out]\n"                                                                                        \
                                                                                                                       \
   /* save m0 in m1 */                                                                                                 \
   "maskand m1, m0, m0\n" /* set m0 to 0 in lanes where y==0*/                                                         \
   "feqm.ps m0, %[y], %[out]\n"                                                                                        \
   "masknot m0, m0\n" /* compute y = y*log(abs(x)) */ /* if y was 0, because of the mask, the result will still be 0*/ \
   "fsgnjx.ps %[x], %[x], %[x]\n"                     /* x = abs(x) */                                                 \
   "flog.ps %[x], %[x]\n"                             /*x=log(abs(x)*/                                                 \
   "fmul.ps %[y], %[y], %[x]\n"                       /* y = y*log(abs(x)) */                                          \
                                                                                                                       \
   /* compute out = exp(y) */                                                                                          \
   "maskand m0, m1, m1\n" /*restore saved mask*/                                                                       \
   "fexp.ps %[out], %[y]\n"                                                                                            \
                                                                                                                       \
   /* at this point:                                                                                                   \
      if x==0 => out=0                                                                                                 \
      if y==0 => out=1                                                                                                 \
      we still need to treat case c                                                                                    \
   */                                                                                                                  \
   "maskand m0, m2, m5\n"               /*x negative, y integer*/                                                      \
   "maskand m0, m0, m3\n"               /*x negative, y integer, y odd*/                                               \
   "fsgnjn.ps %[out], %[out], %[out]\n" /*change the sign*/                                                            \
                                                                                                                       \
   /* set out=NaN for x<0, y=not integer */                                                                            \
   "masknot m2, m2\n"                                                                                                  \
   "maskand m0, m2, m4\n"                                                                                              \
   "fbcx.ps %[out], %[nan]\n" /* and restore the mask for the store*/                                                  \
   "maskand m0, m7, m7\n"                                                                                              \
   : [out] "=&f"(out), [x] "+&f"(in1), [y] "+&f"(in2)                                                                  \
   : [nan] "r"(NAN), [neg_m] "r"(NEG_MASK), [inf] "r"(INFINITY)                                                        \
   : "ft0", "ft1", "ft2", "ft3")

#define EB_Pow_INDEX_COMPUTE ("invalid")

/**
 * @brief Given two tensors, it gives the result of the opType applied elementwise.
 *
 * Given an operator opType and two input tensors A, B, it generates a
 * tensor C in the following way @f$ C_{i,j} = opType(A_{i,j}, B_{i,j}) @f$. 
 * This is the threaded and and vectorized version of the operator.
 * 
 * @note This implementation is similar to the CopyInstVectorized, where the
 *  code is more explained.
 * 
 * @warning It comes without doubt that A and B must have the same dimensions.
 * 
* @note This is a generalization of the previous version as it allows the 
 *  types of the three tensors not being the same.
 *
 * @tparam src1Type The type of the elements in the first input tensors.
 * @tparam src2Type The type of the elements in the second input tensors.
 * @tparam dstType The type of the elements in the output tensor.
 * @tparam opType An operator that takes two srcType elements and returns a 
 *   srcType (+, ·, etc).
 * @param[out] dstT LibTensor pointer to the output matrix.
 * @param[in] in1T LibTensor pointer to the src1 matrix
 * @param[in] in2T LibTensor pointer to the src2 matrix
 * @param[in] flags Controls the active shires and the type of evict that 
 *  should be done at the end of the function.
 */
template <ElemKind elK, typename opType>
INLINE_ATTR void fwdLibElementInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,
                                   const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  static_assert(elK != Int64ITy, "Int64Ty not supported");
  // just return if minion is not to be used

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  using storage_t = typename elemKind2elemTy<elK>::type;
  constexpr auto typeSize = Type::getElementSize(elK);

  // set all lanes active by default
  __asm__ __volatile__("mov.m.x m0, zero, 0xFF \n");

  // compute function
  auto compute = [&](const uintptr_t dstAddr, const uintptr_t src1Addr, uintptr_t src2Addr, const dim_t valid) {
    // set mask
    if (valid < 8) {
      uint8_t mask = static_cast<uint8_t>((1UL << valid) - 1);
      __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
    } else {
      __asm__ __volatile__("mov.m.x m0, zero, 0xFF \n");
    }

    // setup quantization (attribute unused just avoids warnings of variable not being used, which happens if not
    // [de]quantizing)
    float in1Scale __attribute__((unused));
    float in1Offset __attribute__((unused));
    float in2Scale __attribute__((unused));
    float in2Offset __attribute__((unused));
    float outScaleRcp __attribute__((unused));
    float outOffset __attribute__((unused));

    if constexpr (isQuantizedElemKind(elK)) {
      setupDequantize(in1Scale, in1Offset, in1T->getScale(), in1T->getOffset());
      setupDequantize(in2Scale, in2Offset, in2T->getScale(), in2T->getOffset());
      setupQuantize(outScaleRcp, outOffset, outT->getScale(), outT->getOffset());
    }

    // setup memory access
    uint64_t memConf;
    float memIndices, memIndicesHigh;
    setupGatherScatterConfig<typeSize>(memConf, memIndices, memIndicesHigh);

    // load operands
    float in1, in2;
    float in1H, in2H __attribute__((unused));
    load<typeSize>(src1Addr, memConf, memIndices, memIndicesHigh, in1, in1H);
    load<typeSize>(src2Addr, memConf, memIndices, memIndicesHigh, in2, in2H);

    // compute result
    float out = 0.f;
    float outH __attribute__((unused)) = 0.f;

    // compute (only one will apply, because macro has if constexpr(elk ==XX)
    EB_COMPUTE(Add, isIndexElemKind(elK), elK == FloatTy, false);
    EB_COMPUTE(Sub, isIndexElemKind(elK), elK == FloatTy, false);
    EB_COMPUTE(Mul, isIndexElemKind(elK), elK == FloatTy, false);
    EB_COMPUTE(Min, isIndexElemKind(elK), elK == FloatTy, false);
    EB_COMPUTE(Max, isIndexElemKind(elK), elK == FloatTy, false);
    EB_COMPUTE(Div, false, elK == FloatTy, true);  // for div, always compute in float
    EB_COMPUTE(Pow, false, elK == FloatTy, false); // for pow, always compute in float

    // and finally, store
    store<typeSize>(dstAddr, memConf, memIndices, memIndicesHigh, out, outH);
  };

  outT->partitionLoop2<storage_t>(minionId, activeMinions, flags, in1T, in2T, compute);

  // mask back to 0xff
  __asm__ __volatile__("mov.m.x m0, zero, 0xFF \n");
}

////////////////////////////////////////////////////////////////////////////////
// especialization of int64, which is not vectorized
////////////////////////////////////////////////////////////////////////////////
#define EB_I64_COMPUTE(OP_, BODY_)                                                                                     \
  template <>                                                                                                          \
  INLINE_ATTR void fwdLibElementInst<Int64ITy, OP_>(LibTensor * outT, LibTensor * in1T, LibTensor * in2T,              \
                                                    uint64_t flags, const uint32_t minionOffset,                       \
                                                    const uint32_t assignedMinions) {                                  \
    assert(get_minion_id() >= minionOffset);                                                                           \
    size_t minionId = get_minion_id() - minionOffset;                                                                  \
    size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;           \
    if (minionId >= activeMinions)                                                                                     \
      return;                                                                                                          \
                                                                                                                       \
    auto compute = [](uintptr_t outP, uintptr_t in1P, uintptr_t in2P, size_t) {                                        \
      int64_t* o = reinterpret_cast<int64_t*>(outP);                                                                   \
      int64_t a = *(reinterpret_cast<int64_t*>(in1P));                                                                 \
      int64_t b = *(reinterpret_cast<int64_t*>(in2P));                                                                 \
      *o = BODY_;                                                                                                      \
    };                                                                                                                 \
    outT->partitionLoop2<int64_t, int64_t, int64_t, 1>(minionId, static_cast<uint32_t>(activeMinions), flags, in1T,    \
                                                       in2T, compute);                                                 \
  }

EB_I64_COMPUTE(Add, a + b)
EB_I64_COMPUTE(Sub, a - b)
EB_I64_COMPUTE(Div, a / b)
EB_I64_COMPUTE(Mul, a* b)
EB_I64_COMPUTE(Min, std::min(a, b))
EB_I64_COMPUTE(Max, std::max(a, b))

////////////////////////////////////////////////////////////////////////////////
// individual functions per operator (forwarding call to the previous ones with
// the proper parameters)
////////////////////////////////////////////////////////////////////////////////
#define EltWiseBinaryInst(name, opType)                                                                                \
  template <ElemKind elK>                                                                                              \
  INLINE_ATTR void fwdLib##name##Inst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,               \
                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {           \
    inlining::fwdLibElementInst<elK, opType>(outT, in1T, in2T, flags, minionOffset, assignedMinions);                  \
  }

  EltWiseBinaryInst(ElementAdd, Add)
  EltWiseBinaryInst(ElementSub, Sub)
  EltWiseBinaryInst(ElementDiv, Div)
  EltWiseBinaryInst(ElementMul, Mul)
  EltWiseBinaryInst(ElementMin, Min)
  EltWiseBinaryInst(ElementMax, Max)
  EltWiseBinaryInst(ElementPow, Pow)

#undef EltWiseBinaryInst

  } // namespace inlining

  } // namespace dnn_lib

#endif // _ELEMENT_INST_H_
