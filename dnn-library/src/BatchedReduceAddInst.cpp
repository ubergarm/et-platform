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

#include "BatchedReduceAddInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elK>
void fwdLibBatchedReduceAddInst(LibTensor* outT, LibTensor* inT, unsigned int axis) {

  dnn_lib::inlining::fwdLibBatchedReduceAddInst<elK>(outT, inT, axis);
}

template <ElemKind elK>
void fwdLibBatchedReduceAddInstThreaded(LibTensor* outT, LibTensor* inT,
                                        unsigned int axis, uint64_t flags) {

  dnn_lib::inlining::fwdLibBatchedReduceAddInstThreaded<elK>(outT, inT, axis, flags);
}

void fwdLibBatchedReduceAddInstInt8(LibTensor* outT, LibTensor* inT, unsigned int axis) {

  dnn_lib::inlining::fwdLibBatchedReduceAddInstInt8(outT, inT, axis);
}


void fwdLibBatchedReduceAddInstInt8Threaded(LibTensor* outT, LibTensor* inT,
                                            unsigned int axis, uint64_t flags) {

  dnn_lib::inlining::fwdLibBatchedReduceAddInstInt8Threaded(outT, inT, axis, flags);

}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibBatchedReduceAddInst, LibTensor* outT, LibTensor* inT, unsigned int axis);

GEN_INSTANCES_OP(template, fwdLibBatchedReduceAddInstThreaded, LibTensor* outT, LibTensor* inT, unsigned int axis, uint64_t flags);
  
} // namespace dnn_lib
