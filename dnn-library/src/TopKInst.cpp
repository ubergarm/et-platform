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

#include "TopKInst.h" // From include/inlining

namespace dnn_lib {

void partialQuicksort(void *vals, void *inds, int low, int high, int m) {
  if (low < high) {
    int pidx = dnn_lib::inlining::partition(vals, inds, low, high);
    partialQuicksort(vals, inds, low, pidx - 1, m);
    if (pidx < m) {
      partialQuicksort(vals, inds, pidx + 1, high, m);
    }
  }
}

template <typename srcType>
void fwdLibTopKInst(void *dstT, void *dstDims, void *dstPitches,
                             void *dstT2, void *dst2Dims, void *dst2Pitches,
                             void *srcT, void *srcDims, void *srcPitches,
                             unsigned int srcDimNum, unsigned int k,
                             const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibTopKInst<srcType>(dstT, dstDims, dstPitches,
                             dstT2, dst2Dims, dst2Pitches,
                             srcT, srcDims, srcPitches,
                             srcDimNum, k,
                             scale, offset);
}

template <typename srcType>
void fwdLibTopKInstThreaded_all(
    void *dstT, void *dstDims, void *dstPitches, void *dstT2, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, unsigned int k, const float *scale, const int32_t *offset,
    uint64_t flags) {

  dnn_lib::inlining::fwdLibTopKInstThreaded_all<srcType>(
    dstT, dstDims, dstPitches, dstT2, dst2Dims,
    dst2Pitches, srcT, srcDims, srcPitches,
    srcDimNum, k, scale, offset,
    flags);
}

template <typename srcType>
void fwdLibTopKInstThreaded_k4(
    void *dstT, void *dstDims, void *dstPitches, void *dstT2, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, unsigned int k, const float *scale, const int32_t *offset,
    uint64_t flags) {

  dnn_lib::fwdLibTopKInstThreaded_k4<srcType>(
    dstT, dstDims, dstPitches, dstT2, dst2Dims,
    dst2Pitches, srcT, srcDims, srcPitches,
    srcDimNum, k, scale, offset,
    flags);
}

template <typename srcType>
void fwdLibTopKInstThreaded_k8(
    void *dstT, void *dstDims, void *dstPitches, void *dstT2, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, unsigned int k, const float *scale, const int32_t *offset,
    uint64_t flags) {

  dnn_lib::inlining::fwdLibTopKInstThreaded_k8<srcType>(
    dstT, dstDims, dstPitches, dstT2, dst2Dims,
    dst2Pitches, srcT, srcDims, srcPitches,
    srcDimNum, k, scale, offset,
    flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibTopKInst, void *dstT, void *dstDims, void *dstPitches, void *dstT2,
                              void *dst2Dims, void *dst2Pitches,void *srcT, void *srcDims, void *srcPitches,
                              unsigned int srcDimNum, unsigned int k,
                              const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibTopKInstThreaded_all, void *dstT, void *dstDims, void *dstPitches, void *dstT2,
                              void *dst2Dims, void *dst2Pitches,void *srcT, void *srcDims, void *srcPitches,
                              unsigned int srcDimNum, unsigned int k,
                              const float *scale, const int32_t *offset, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibTopKInstThreaded_k4, void *dstT, void *dstDims, void *dstPitches, void *dstT2,
                              void *dst2Dims, void *dst2Pitches,void *srcT, void *srcDims, void *srcPitches,
                              unsigned int srcDimNum, unsigned int k,
                              const float *scale, const int32_t *offset, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibTopKInstThreaded_k8, void *dstT, void *dstDims, void *dstPitches, void *dstT2,
                              void *dst2Dims, void *dst2Pitches,void *srcT, void *srcDims, void *srcPitches,
                              unsigned int srcDimNum, unsigned int k,
                              const float *scale, const int32_t *offset, uint64_t flags);
} // namespace dnn_lib
