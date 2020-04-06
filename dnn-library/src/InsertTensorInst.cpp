/*-------------------------------------------------------------------------
 * Copyright (C) 2020, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include "InsertTensorInst.h"

namespace dnn_lib {

template <typename srcType>
void fwdLibInsertTensorInst(void *dst, void *dstDims, void *dstPitches,
                                     unsigned int dstDimNum, void *src2,
                                     void *src2Dims, void *src2Pitches,
                                     void *pcoord, unsigned int count,
                                     unsigned int axis, const float *scale,
                                     const int32_t *offset, uint64_t flags,
                                     const uint32_t minionOffset) {

  dnn_lib::inlining::fwdLibInsertTensorInst<srcType>(dst, dstDims, dstPitches,
                                     dstDimNum, src2,
                                     src2Dims, src2Pitches,
                                     pcoord, count,
                                     axis, scale,
                                     offset, flags,
                                     minionOffset);
}

template <typename srcType>
void fwdLibInsertTensorInstThreaded(void *dst, void *dstDims,
                                             void *dstPitches,
                                             unsigned int dstDimNum, void *src2,
                                             void *src2Dims, void *src2Pitches,
                                             void *poffsets, unsigned int count,
                                             unsigned int axis, const float *scale,
                                             const int32_t *offset, uint64_t flags,
                                             const uint32_t minionOffset,
                                             const  uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibInsertTensorInstThreaded<srcType>(dst, dstDims,
                                             dstPitches,
                                             dstDimNum,src2,
                                             src2Dims, src2Pitches,
                                             poffsets, count,
                                             axis, scale,
                                             offset, flags,
                                             minionOffset,
                                             assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibInsertTensorInst, void *dst, void *dstDims,
                 void *dstPitches, unsigned int dstDimNum, void *src2, 
                 void *src2Dims, void *src2Pitches, void * poffsets, 
                 unsigned int count, unsigned int axis, const float *scale, 
                 const int32_t *offset, uint64_t flags, const uint32_t minionOffset);

GEN_INSTANCES_OP(template, fwdLibInsertTensorInstThreaded, void *dst, void *dstDims,
                 void *dstPitches, unsigned int dstDimNum, void *src2, 
                 void *src2Dims, void *src2Pitches, void * poffsets, 
                 unsigned int count, unsigned int axis, const float *scale, 
                 const int32_t *offset, uint64_t flags,
                 const uint32_t minionOffset, const uint32_t assignedMinions);

} // namespace dnn_lib
