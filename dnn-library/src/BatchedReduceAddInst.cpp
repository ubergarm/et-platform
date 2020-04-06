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

#include "BatchedReduceAddInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibBatchedReduceAddInst(void *pdst, void *pdstDims,
                                         void *pdstPitches, void *pbatch,
                                         void *pbatchDims, void *pbatchPitches,
                                         unsigned int pbatchDimNum,
                                         unsigned int axis, const float *scale,
                                         const int32_t *offset) {

  dnn_lib::inlining::fwdLibBatchedReduceAddInst<srcType>(pdst, pdstDims,
                                         pdstPitches, pbatch,
                                         pbatchDims, pbatchPitches,
                                         pbatchDimNum,
                                         axis, scale,
                                         offset);
}

template <typename srcType>
void fwdLibBatchedReduceAddInstThreaded(void *pdst, void *pdstDims,
                                                 void *pdstPitches, void *pbatch,
                                                 void *pbatchDims, void *pbatchPitches,
                                                 unsigned int pbatchDimNum,
                                                 unsigned int axis, const float *scale,
                                                 const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibBatchedReduceAddInstThreaded<srcType>(pdst, pdstDims,
                                                 pdstPitches, pbatch,
                                                 pbatchDims, pbatchPitches,
                                                 pbatchDimNum,
                                                 axis, scale,
                                                 offset, flags);
}

void fwdLibBatchedReduceAddInstInt8(
    void *pdst, void *pdstDims, void *pdstPitches, void *pbatch,
    void *pbatchDims, void *pbatchPitches, unsigned int pbatchDimNum,
    unsigned int axis, const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibBatchedReduceAddInstInt8(
    pdst, pdstDims, pdstPitches, pbatch,
    pbatchDims, pbatchPitches, pbatchDimNum,
    axis, scale, offset);
}


void fwdLibBatchedReduceAddInstInt8Threaded(
    void *pdst, void *pdstDims, void *pdstPitches, void *pbatch,
    void *pbatchDims, void *pbatchPitches, unsigned int pbatchDimNum,
    unsigned int axis, const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibBatchedReduceAddInstInt8Threaded(
    pdst, pdstDims, pdstPitches, pbatch,
    pbatchDims, pbatchPitches, pbatchDimNum,
    axis, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibBatchedReduceAddInst, void *pdst, void *pdstDims, void *pdstPitches,
                                   void *pbatch, void *pbatchDims, void *pbatchPitches,
                                   unsigned int pbatchDimNum, unsigned int axis,
                                   const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibBatchedReduceAddInstThreaded, void *pdst, void *pdstDims, void *pdstPitches,
                                   void *pbatch, void *pbatchDims, void *pbatchPitches,
                                   unsigned int pbatchDimNum, unsigned int axis,
                                   const float *scale, const int32_t *offset, uint64_t flags);
} // namespace dnn_lib
