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

#include "ElementSelectInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibElementSelectInst(
    void *dstT, void *dstDims, void *dstPitches, void *condT, void *condDims,
    void *condPitches, void *srcT1, void *srcDims, void *src1Pitches,
    unsigned int srcDimNum, void *srcT2, void *src2Pitches,
    const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibElementSelectInst<srcType>(
    dstT, dstDims, dstPitches, condT, condDims,
    condPitches, srcT1, srcDims, src1Pitches,
    srcDimNum, srcT2, src2Pitches,
    scale, offset);
}

template <typename srcType>
void fwdLibElementSelectInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *condT, void *condDims,
    void *condPitches, void *srcT1, void *srcDims, void *src1Pitches,
    unsigned int srcDimNum, void *srcT2, void *src2Pitches, const float *scale, const int32_t *offset,
    uint64_t flags) {

  dnn_lib::inlining::fwdLibElementSelectInstThreaded<srcType>(
    dstT, dstDims, dstPitches, condT, condDims,
    condPitches, srcT1, srcDims, src1Pitches,
    srcDimNum, srcT2, src2Pitches, scale, offset,
    flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibElementSelectInst,void *dstT, void *dstDims,
                                 void *dstPitches, void *condT,
                                 void *condDims, void *condPitches,
                                 void *srcT1, void *srcDims,
                                 void *src1Pitches,unsigned int srcDimNum,
                                 void *srcT2, void *src2Pitches, const float * scale,
                                 const int32_t * offset);

GEN_INSTANCES_OP(template, fwdLibElementSelectInstThreaded,void *dstT, void *dstDims,
                                        void *dstPitches, void *condT,
                                        void *condDims, void *condPitches,
                                        void *srcT1, void *srcDims,
                                        void *src1Pitches,unsigned int srcDimNum,
                                        void *srcT2, void *src2Pitches, const float * scale,
                                        const int32_t * offset, uint64_t flags);


} // namespace dnn_lib

