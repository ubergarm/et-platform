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

#include "ElementSingleInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType, typename opType>
void fwdLibElementSingleInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *srcT1,
                                      void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, const float *scale,
                                      const int32_t *offset) {

  dnn_lib::inlining::fwdLibElementSingleInst<srcType, opType>(dstT, dstDims,
                                      dstPitches, srcT1,
                                      srcDims, srcPitches,
                                      srcDimNum, scale,
                                      offset);
}

template <typename srcType, typename opType>
void fwdLibElementSingleInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, const float *scale,
    const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementSingleInstThreaded<srcType, opType>(
    dstT, dstDims, dstPitches, srcT1, srcDims,
    srcPitches, srcDimNum, scale,
    offset, flags);
}

template <typename src1Type, typename dstType, typename opType>
void fwdLibElementSingleInstVectorized(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, const float *scale,
    const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementSingleInstVectorized<src1Type, dstType, opType>(
    dstT, dstDims, dstPitches, srcT1, srcDims,
    srcPitches, srcDimNum, scale,
    offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_INSTANCES(template, fwdLibElementSingleInst,ElementLog,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum,
                                 const float * scale, const int32_t * offset);

GEN_INSTANCES_INSTANCES(template, fwdLibElementSingleInstThreaded,ElementLog,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum,
                                 const float * scale, const int32_t * offset, uint64_t flags);

GEN_INSTANCES_2TYPE(template, fwdLibElementSingleInstVectorized,ElementLog,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum,
                                 const float * scale, const int32_t * offset, uint64_t flags);

} // namespace dnn_lib
