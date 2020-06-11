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

#include "BatchedAddInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elK>
void fwdLibBatchedAddInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T) {

  dnn_lib::inlining::fwdLibBatchedAddInst<elK>(outT, in1T, in2T);
}

template <ElemKind elK>
void fwdLibBatchedAddInstThreaded(LibTensor* outT, LibTensor* in1T,
                                  LibTensor* in2T, uint64_t flags) {

  dnn_lib::inlining::fwdLibBatchedAddInstThreaded<elK>(outT, in1T,
                                                           in2T, flags);
}

// TODO: Special implementation to support int8_t + int32_t sum, as a quick fix
// we implement this function to support it, the correct way is extend the
// BatchedAdd templatized op and the Operator class in order to support 2
// different templates
void fwdLibBatchedAddInsti8i32(LibTensor* outT, LibTensor* in1T, LibTensor* in2T) {

  dnn_lib::inlining::fwdLibBatchedAddInsti8i32(outT, in1T, in2T);
}

void fwdLibBatchedAddInsti8i32Threaded(LibTensor* outT, LibTensor* in1T,
                                       LibTensor* in2T, uint64_t flags) {

  dnn_lib::inlining::fwdLibBatchedAddInsti8i32Threaded(outT, in1T, in2T, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibBatchedAddInst, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_OP(template, fwdLibBatchedAddInstThreaded, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, uint64_t flags);
} // namespace dnn_lib
