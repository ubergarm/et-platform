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

#include "SparseToDenseInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elK>
void fwdLibSparseToDenseInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T) {

  dnn_lib::inlining::fwdLibSparseToDenseInst<elK>(outT, in1T, in2T);
}

template <ElemKind elK>
void fwdLibSparseToDenseInstThreaded(LibTensor* outT, LibTensor* in1T,
                                     LibTensor* in2T, uint64_t flags) {

  dnn_lib::inlining::fwdLibSparseToDenseInstThreaded<elK>(outT, in1T, in2T, flags);
}

template <ElemKind elK>
void fwdLibSparseToDenseInstVectorized(LibTensor* outT, LibTensor* in1T,
                                       LibTensor* in2T, const float* scale,
                                       const int32_t* offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibSparseToDenseInstVectorized<elK>(outT, in1T, in2T, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSparseToDenseInst, LibTensor* outT, LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_OP(template, fwdLibSparseToDenseInstThreaded, LibTensor* outT, LibTensor* in1T,
                 LibTensor* in2T, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibSparseToDenseInstVectorized, LibTensor* outT, LibTensor* in1T,
                 LibTensor* in2T, const float* scale, const int32_t* offset, uint64_t flags);

} // namespace dnn_lib
