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

#include "CrossEntropyLossInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibCrossEntropyLossInst(void *dstT, void *srcT, void *srcDims,
                                         void *srcPitches,
                                         unsigned int srcDimNum, void *labelsT,
                                         const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibCrossEntropyLossInst<srcType>(dstT, srcT, srcDims,
                                         srcPitches,
                                         srcDimNum, labelsT,
                                         scale, offset);
}

template <typename srcType>
void fwdLibCrossEntropyLossInstThreaded(
    void *dstT, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, void *labelsT, const float *scale, const int32_t *offset,
    uint64_t flags) {

  dnn_lib::inlining::fwdLibCrossEntropyLossInstThreaded<srcType>(
    dstT, srcT, srcDims, srcPitches,
    srcDimNum, labelsT, scale, offset,
    flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibCrossEntropyLossInst, void *dstT, void *srcT, void *srcDims,
                                   void *srcPitches, unsigned int srcDimNum, void* labelsT,
                                   const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibCrossEntropyLossInstThreaded, void *dstT, void *srcT, void *srcDims,
                                   void *srcPitches, unsigned int srcDimNum, void* labelsT,
                                   const float *scale, const int32_t *offset, uint64_t flags);
} // namespace dnn_lib
