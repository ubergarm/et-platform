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

#include "FullyConnectedInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibFullyConnectedInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                              LibTensor* in3T) {

  dnn_lib::inlining::fwdLibFullyConnectedInst<srcType>(outT, in1T, in2T, in3T);
}

template <typename srcType>
void fwdLibFullyConnectedInstThreaded(LibTensor* outT, LibTensor* in1T,
                                      LibTensor* in2T, LibTensor* in3T,
                                      uint64_t flags) {

  dnn_lib::inlining::fwdLibFullyConnectedInstThreaded<srcType>(outT, in1T, in2T,
                                                               in3T, flags);
}

template <typename src1Type, typename src2Type, typename dstType>
void fwdLibFullyConnectedInstVectorized(LibTensor* outT, LibTensor* in1T,
                                        LibTensor* in2T, LibTensor* in3T,
                                        const float* scale, const int32_t* offset,
                                        uint64_t flags) {

  dnn_lib::inlining::fwdLibFullyConnectedInstVectorized<src1Type, src2Type, dstType>(
                                         outT, in1T, in2T, in3T, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibFullyConnectedInst, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, LibTensor* in3T);

GEN_INSTANCES_OP(template, fwdLibFullyConnectedInstThreaded, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, uint64_t flags);

GEN_INSTANCES_3TYPE_OP(template, fwdLibFullyConnectedInstVectorized, LibTensor* outT,
                       LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                       const float* scale, const int32_t* offset, uint64_t flags);

} // namespace dnn_lib
