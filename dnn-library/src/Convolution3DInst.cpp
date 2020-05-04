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

#include "Convolution3DInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibConvolution3DInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                             LibTensor* in3T, void *pkernels, void *pstrides,
                             void *ppads, unsigned int group) {

  dnn_lib::inlining::fwdLibConvolution3DInst<srcType>(outT, in1T, in2T, in3T,
                                              pkernels, pstrides, ppads, group);
}

template <typename srcType>
void fwdLibConvolution3DInstThreaded(LibTensor* outT, LibTensor* in1T,
                                     LibTensor* in2T, LibTensor* in3T,
                                     void *pkernels, void *pstrides,
                                     void *ppads, unsigned int group,
                                     uint64_t flags) {

  dnn_lib::inlining::fwdLibConvolution3DInstThreaded<srcType>(outT, in1T, in2T,
                                                              in3T, pkernels,
                                                              pstrides, ppads,
                                                              group, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibConvolution3DInst, LibTensor* outT, LibTensor* in1T,
                 LibTensor* in2T, LibTensor* in3T, void *pkernels, void *pstrides,
                 void *ppads, unsigned int group);
GEN_INSTANCES_OP(template, fwdLibConvolution3DInstThreaded, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                 void *pkernels, void *pstrides, void *ppads, unsigned int group,
                 uint64_t flags);

} // namespace dnn_lib
