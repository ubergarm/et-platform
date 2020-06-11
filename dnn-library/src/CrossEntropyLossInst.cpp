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

#include "CrossEntropyLossInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elK>
void fwdLibCrossEntropyLossInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T) {

  dnn_lib::inlining::fwdLibCrossEntropyLossInst<elK>(outT, in1T, in2T);
}

template <ElemKind elK>
void fwdLibCrossEntropyLossInstThreaded(LibTensor* outT, LibTensor* in1T,
                                        LibTensor* in2T, uint64_t flags) {

  dnn_lib::inlining::fwdLibCrossEntropyLossInstThreaded<elK>(outT, in1T, in2T,
                                                                 flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibCrossEntropyLossInst, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T );

GEN_INSTANCES_OP(template, fwdLibCrossEntropyLossInstThreaded, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T , uint64_t flags);
} // namespace dnn_lib
