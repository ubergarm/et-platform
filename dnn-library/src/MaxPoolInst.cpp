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

#include "MaxPoolInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibMaxPoolInst(bool argMax, LibTensor* outT, LibTensor* out2T,
                       LibTensor* inT, void *srcMatrixPitchesNoPadding,
                       void *pkernels, void *pstrides, void *ppads) {

  dnn_lib::inlining::fwdLibMaxPoolInst<srcType>(argMax, outT, out2T, inT,
                                                srcMatrixPitchesNoPadding,
                                                pkernels, pstrides, ppads);
}

template <typename srcType, typename dstType>
void fwdLibMaxPoolInstThreaded(bool argMax, LibTensor* outT, LibTensor* out2T,
                               LibTensor* inT, void *srcMatrixPitchesNoPadding,
                               void *pkernels, void *pstrides, void *ppads,
                               uint64_t flags) {

  dnn_lib::inlining::fwdLibMaxPoolInstThreaded<srcType, dstType>(argMax, outT,
                                         out2T, inT, srcMatrixPitchesNoPadding,
                                         pkernels, pstrides, ppads, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibMaxPoolInst, bool argMax, LibTensor* outT,
                 LibTensor* out2T, LibTensor* inT,
                 void *srcMatrixPitchesNoPadding, void *pkernels,
                 void *pstrides, void *ppads);

GEN_INSTANCES_2TYPE_OP(template, fwdLibMaxPoolInstThreaded, bool argMax,
                       LibTensor* outT, LibTensor* out2T, LibTensor* inT,
                       void *srcMatrixPitchesNoPadding, void *pkernels,
                       void *pstrides, void *ppads, uint64_t flags);
} // namespace dnn_lib
