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

#include "BatchOneHotInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibBatchOneHotInst(LibTensor* outT, LibTensor* in1T,
                           LibTensor* in2T, LibTensor* in3T) {

  dnn_lib::inlining::fwdLibBatchOneHotInst<srcType>(outT, in1T, in2T, in3T);
}

template <typename srcType>
void fwdLibBatchOneHotInstThreaded(LibTensor* outT, LibTensor* in1T,
                                   LibTensor* in2T, LibTensor* in3T,
                                   uint64_t flags) {

  dnn_lib::inlining::fwdLibBatchOneHotInstThreaded<srcType>(outT, in1T,
                                                            in2T, in3T,
                                                             flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibBatchOneHotInst, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, LibTensor* in3T);

GEN_INSTANCES_OP(template, fwdLibBatchOneHotInstThreaded, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                 uint64_t flags);
}
