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

template <typename srcType, typename indexType>
void fwdLibGatherInst(void *dstT, void *dstDims, void *dstPitches,
                               void *srcT, void *srcDims, void *srcPitches,
                               unsigned int srcDimsNum, void *indexT,
                               void *indicesDims, void *pindicesPitches,
                               unsigned int batchedDims, const float *scale,
                               const int32_t *offset) {

  dnn_lib::inlining::fwdLibGatherInst<srcType, indexType>(dstT, dstDims, dstPitches,
                               srcT, srcDims, srcPitches,
                               srcDimsNum, indexT,
                               indicesDims, pindicesPitches,
                               batchedDims, scale,
                               offset);
}

template <typename srcType, typename indexType>
void fwdLibGatherInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT, void *srcDims,
    void *srcPitches, unsigned int srcDimsNum, void *indexT, void *indicesDims,
    void *pindicesPitches, unsigned int indicesDimsNum,
    unsigned int batchedDims, // indicesDimsNum is an new parameter for the threaded version.
    const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibGatherInstThreaded<srcType, indexType>(
    dstT, dstDims, dstPitches, srcT, srcDims,
    srcPitches, srcDimsNum, indexT, indicesDims,
    pindicesPitches, indicesDimsNum,
    batchedDims,
    scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP_INDEX(template, fwdLibGatherInst, void *dstT, void *dstDims, void *dstPitches,
                               void *srcT, void *srcDims, void *srcPitches,
                               unsigned int srcDimsNum, void *indexT, void *indicesDims,
                               void *pindicesPitches, unsigned int batchedDims,
                               const float *scale, const int32_t *offset);

GEN_INSTANCES_OP_INDEX(template, fwdLibGatherInstThreaded, void *dstT, void *dstDims, void *dstPitches,
                               void *srcT, void *srcDims, void *srcPitches,
                               unsigned int srcDimsNum, void *indexT, void *indicesDims,
                               void *pindicesPitches, unsigned int indicesDimsNum,
                               unsigned int batchedDims, const float *scale, const int32_t *offset, uint64_t flags);
} // namespce dnn_lib
