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

#include "GatherInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elK, typename indexType>
void fwdLibGatherInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                      unsigned int batchedDims) {

  dnn_lib::inlining::fwdLibGatherInst<elK, indexType>(outT, in1T, in2T,
                                                          batchedDims);
}

template <ElemKind elK, typename indexType>
void fwdLibGatherInstThreaded(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                              unsigned int batchedDims, uint64_t flags) {

  dnn_lib::inlining::fwdLibGatherInstThreaded<elK, indexType>(outT, in1T, in2T,
                                                                  batchedDims, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP_INDEX(template, fwdLibGatherInst, LibTensor* outT, LibTensor* in1T,
                       LibTensor* in2T, unsigned int batchedDims);

GEN_INSTANCES_OP_INDEX(template, fwdLibGatherInstThreaded, LibTensor* outT, LibTensor* in1T,
                       LibTensor* in2T, unsigned int batchedDims,  uint64_t flags);
} // namespce dnn_lib
