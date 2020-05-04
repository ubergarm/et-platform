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

#include "SparseLengthsWeightedSumInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibSparseLengthsWeightedSumInst(LibTensor* outT, LibTensor* in1T,
                                        LibTensor* in2T, LibTensor* in3T,
                                        LibTensor* in4T,
                                        unsigned int pLengthsSize) {

  dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInst<srcType>(outT, in1T,
                                                                 in2T, in3T,
                                                                 in4T,
                                                                 pLengthsSize);
}

template <typename srcType>
void fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* outT, LibTensor* in1T,
                                                LibTensor* in2T, LibTensor* in3T,
                                                LibTensor* in4T,
                                                unsigned int pLengthsSize,
                                                uint64_t flags) {

  dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInstThreaded<srcType>(outT, in1T,
                                                                         in2T, in3T,
                                                                         in4T,
                                                                         pLengthsSize,
                                                                         flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSparseLengthsWeightedSumInst, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                 LibTensor* in4T, unsigned int pLengthsSize);

GEN_INSTANCES_OP(template, fwdLibSparseLengthsWeightedSumInstThreaded, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
                 unsigned int pLengthsSize, uint64_t flags);

} // namespace dnn_lib
