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

#include "MatMulInstTransposed.h"

namespace dnn_lib {

template <typename srcType>
void fwdLibMatMulInstTransposed(LibTensor* outT, LibTensor* in1T,
                                LibTensor* in2T) {

  dnn_lib::inlining::fwdLibMatMulInstTransposed<srcType>(outT, in1T, in2T);
}

template <typename srcType>
void fwdLibMatMulInstThreadedTransposed(LibTensor* outT, LibTensor* in1T,
                                        LibTensor* in2T, uint64_t flags) {

  dnn_lib::inlining::fwdLibMatMulInstThreadedTransposed<srcType>(outT, in1T,
                                                                 in2T, flags);
}

template <typename srcType>
void fwdLibMatMulInstVectorizedTransposed(LibTensor* outT, LibTensor* in1T,
                                          LibTensor* in2T, uint64_t flags) {

  dnn_lib::inlining::fwdLibMatMulInstVectorizedTransposed<srcType>(outT, in1T,
                                                                   in2T,
                                                                   flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibMatMulInstTransposed, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_OP(template, fwdLibMatMulInstThreadedTransposed, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibMatMulInstVectorizedTransposed, LibTensor* out,
                 LibTensor* in1T, LibTensor* in2T, uint64_t flags);

} // namespace dnn_lib
