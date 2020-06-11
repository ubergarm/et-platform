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

#include "MatMulInst.h" // From include/inlining path

namespace dnn_lib {

template <ElemKind elK>
void fwdLibMatMulInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T) {

  dnn_lib::inlining::fwdLibMatMulInst<elK>(outT, in1T, in2T);
}

template <ElemKind elK>
void fwdLibMatMulInstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, 
                              uint64_t flags, const uint32_t minionOffset,
                              const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibMatMulInstThreaded<elK>(outT, in1T, in2T,  flags,
                                                       minionOffset, assignedMinions);
}


template <ElemKind elK>
void fwdLibMatMulInstVectorized(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, 
                                uint64_t flags, const uint32_t minionOffset,
                                const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibMatMulInstVectorized<elK>(outT, in1T, in2T, flags,
                                                         minionOffset, assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibMatMulInst, LibTensor* inT, LibTensor* weighT,
                 LibTensor* outT);

GEN_INSTANCES_OP(template, fwdLibMatMulInstThreaded, LibTensor* outT, LibTensor* in1T,
                 LibTensor* in2T,  uint64_t flags,
                 const uint32_t minionOffset = 0, const uint32_t numShires = 0);

GEN_INSTANCES_OP(template, fwdLibMatMulInstVectorized, LibTensor* outT, LibTensor* in1T,
                 LibTensor* in2T, uint64_t flags,
                 const uint32_t minionOffset = 0, const uint32_t numShires = 0);

} // namespace dnn_lib
