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

#include "ExtractTensorInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibExtractTensorInst(LibTensor* outT, LibTensor* inT, void *pcoord) {

  dnn_lib::inlining::fwdLibExtractTensorInst<srcType>(outT, inT, pcoord);
}

template <typename srcType>
void fwdLibExtractTensorInstThreaded(LibTensor* outT, LibTensor* inT,
                                     void *pcoord, uint64_t flags) {

  dnn_lib::inlining::fwdLibExtractTensorInstThreaded<srcType>(outT, inT,
                                                              pcoord, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibExtractTensorInst, LibTensor* outT,
                 LibTensor* inT, void * poffsets);

  GEN_INSTANCES_OP(template, fwdLibExtractTensorInstThreaded, LibTensor* outT,
                   LibTensor* inT, void * poffsets, uint64_t flags);

} // namespace dnn_lib
