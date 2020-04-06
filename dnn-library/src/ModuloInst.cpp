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

#include "ModuloInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibModuloInst(void *dstT, void *dstDims, void *dstPitches,
                               void *srcT, void *srcDims, void *srcPitches,
                               unsigned int srcDimNum, long long divisor,
                               bool signFollowDivisor, const float *scale,
                               const int32_t *offset) {

  dnn_lib::inlining::fwdLibModuloInst<srcType>(dstT, dstDims, dstPitches,
                               srcT, srcDims, srcPitches,
                               srcDimNum, divisor,
                               signFollowDivisor, scale,
                               offset);
}

template <typename srcType>
void fwdLibModuloInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, long long divisor,
    bool signFollowDivisor, const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibModuloInstThreaded<srcType>(
    dstT, dstDims, dstPitches, srcT, srcDims,
    srcPitches, srcDimNum, divisor,
    signFollowDivisor, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_INTONLY_OP(template, fwdLibModuloInst, void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, long long divisor, bool signFollowDivisor,
                                 const float * scale, const int32_t * offset);

GEN_INSTANCES_INTONLY_OP(template, fwdLibModuloInstThreaded, void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, long long divisor, bool signFollowDivisor,
                                 const float * scale, const int32_t * offset, uint64_t flags);
} // namespace dnn_lib
