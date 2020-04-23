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

#include "AvgPoolInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibAvgPoolInst(LibTensor* outT, LibTensor* inT, void *pkernels,
                       void *pstrides, void *ppads) {

  dnn_lib::inlining::fwdLibAvgPoolInst<srcType>(outT, inT, pkernels, pstrides,
                                                ppads);
}

template <typename srcType, typename dstType>
void fwdLibAvgPoolInstThreaded(LibTensor* outT, LibTensor* inT,
                               void *pkernels, void *pstrides, void *ppads,
                               uint64_t flags) {

  dnn_lib::inlining::fwdLibAvgPoolInstThreaded<srcType, dstType>(outT, inT,
                                              pkernels, pstrides, ppads, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibAvgPoolInst, LibTensor* outT, LibTensor* inT,
                 void *pkernels, void *pstrides, void *ppads);

GEN_INSTANCES_2TYPE_OP(template, fwdLibAvgPoolInstThreaded, LibTensor* outT,
                       LibTensor* inT, void *pkernels, void *pstrides,
                       void *ppads, uint64_t flags);

} // namespace dnn_lib
