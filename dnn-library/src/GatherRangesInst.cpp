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

#include "GatherRangesInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType, typename indexType>
void fwdLibGatherRangesInst(
    void *dstT, void *dstDims, void *dstPitches, void *dst2T, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimsNum, void *prangesT, void *prangesDims,
    void *prangesPitches, const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibGatherRangesInst<srcType, indexType>(
    dstT, dstDims, dstPitches, dst2T, dst2Dims,
    dst2Pitches, srcT, srcDims, srcPitches,
    srcDimsNum, prangesT, prangesDims,
    prangesPitches, scale, offset);
}


template <typename srcType, typename indexType>
void fwdLibGatherRangesInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *dst2T, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimsNum, void *prangesT, void *prangesDims,
    void *prangesPitches, const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibGatherRangesInstThreaded<srcType, indexType>(
    dstT, dstDims, dstPitches, dst2T, dst2Dims,
    dst2Pitches, srcT, srcDims, srcPitches,
    srcDimsNum, prangesT, prangesDims,
    prangesPitches, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP_INDEX(template, fwdLibGatherRangesInst, void *dstT, void *dstDims, void *dstPitches,
                                     void *dst2T, void *dst2Dims, void *dst2Pitches,
                                     void *srcT, void *srcDims, void *srcPitches,
                                     unsigned int srcDimsNum, void *prangesT, void *prangesDims,
                                     void *prangesPitches, const float *scale, const int32_t *offset);

GEN_INSTANCES_OP_INDEX(template, fwdLibGatherRangesInstThreaded, void *dstT, void *dstDims, void *dstPitches,
                                     void *dst2T, void *dst2Dims, void *dst2Pitches,
                                     void *srcT, void *srcDims, void *srcPitches,
                                     unsigned int srcDimsNum, void *prangesT, void *prangesDims,
                                     void *prangesPitches, const float *scale, const int32_t *offset, uint64_t flags);
} // namespace dnn_lib
