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

#include "TensorViewInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibTensorViewInst(void *dst, void *dstDims, void *dstPitches,
                                   unsigned int dstDimNum, void *src,
                                   void *srcDims, void *srcPitches,
                                   unsigned int srcDimNum, void *pcoord,
                                   const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibTensorViewInst<srcType>(dst, dstDims, dstPitches,
                                   dstDimNum, src,
                                   srcDims, srcPitches,
                                   srcDimNum, pcoord,
                                   scale, offset);
}

template <typename srcType>
void fwdLibTensorViewInstThreaded(
    void *dst, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src, void *srcDims, void *srcPitches, unsigned int srcDimNum,
    void *pcoord, const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibTensorViewInstThreaded<srcType>(
    dst, dstDims, dstPitches, dstDimNum,
    src, srcDims, srcPitches, srcDimNum,
    pcoord, scale, offset, flags);
}

template <typename srcType>
void fwdLibTensorViewInstVectorized(
    void *dst, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src, void *srcDims, void *srcPitches, unsigned int srcDimNum,
    void *pcoord, const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibTensorViewInstVectorized<srcType>(
    dst, dstDims, dstPitches, dstDimNum,
    src, srcDims, srcPitches, srcDimNum,
    pcoord, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibTensorViewInst, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, void *src, void *srcDims,
                             void *srcPitches, unsigned int srcDimNum, void *poffsets,
                             const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibTensorViewInstThreaded, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, void *src, void *srcDims,
                             void *srcPitches, unsigned int srcDimNum, void *poffsets,
                             const float *scale, const int32_t *offset, uint64_t flags );

GEN_INSTANCES_OP(template, fwdLibTensorViewInstVectorized, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, void *src, void *srcDims,
                             void *srcPitches, unsigned int srcDimNum, void *poffsets,
                             const float *scale, const int32_t *offset, uint64_t flags );
} // namespace dnn_lib
