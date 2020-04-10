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

#include "ElementBoolInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType, typename opType>
void fwdLibElementBoolInst(void *dstT, void *dstDims, void *dstPitches,
                                    void *srcT1, void *srcDims,
                                    void *src1Pitches, unsigned int srcDimNum,
                                    void *srcT2, void *src2Pitches, const float *scale,
                                    const int32_t *offset) {

  dnn_lib::inlining::fwdLibElementBoolInst<srcType, opType>(dstT, dstDims, dstPitches,
                                    srcT1, srcDims,
                                    src1Pitches, srcDimNum,
                                    srcT2, src2Pitches, scale,
                                    offset);
}

template <typename srcType, typename opType>
void fwdLibElementBoolInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *src1Pitches, unsigned int srcDimNum, void *srcT2, void *src2Pitches,
    const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementBoolInstThreaded<srcType, opType>(
    dstT, dstDims, dstPitches, srcT1, srcDims,
    src1Pitches, srcDimNum, srcT2, src2Pitches,
    scale, offset, flags);
}

template <typename src1Type, typename src2Type, typename opType>
void fwdLibElementBoolInstVectorized(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *src1Pitches, unsigned int srcDimNum, void *srcT2, void *src2Pitches,
    const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementBoolInstVectorized<src1Type, src2Type, opType>(
    dstT, dstDims, dstPitches, srcT1, srcDims,
    src1Pitches, srcDimNum, srcT2, src2Pitches,
    scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInst,CmpEQ,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,
                                 const float * scale, const int32_t * offset);

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInst,CmpLTE,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,
                                 const float * scale, const int32_t * offset);

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInst,CmpLT,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,
                                 const float * scale, const int32_t * offset);

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInstThreaded,CmpEQ,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,
                                 const float * scale, const int32_t  * offset, uint64_t flags);

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInstThreaded,CmpLTE,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,
                                 const float * scale, const int32_t * offset, uint64_t flags);

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInstThreaded,CmpLT,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,
                                 const float * scale, const int32_t * offset, uint64_t flags);


GEN_INSTANCES_2TYPE(template, fwdLibElementBoolInstVectorized,CmpEQ,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,
                                 const float * scale, const int32_t * offset, uint64_t flags);

GEN_INSTANCES_2TYPE(template, fwdLibElementBoolInstVectorized,CmpLTE,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,
                                 const float * scale, const int32_t * offset, uint64_t flags);
  
GEN_INSTANCES_2TYPE(template, fwdLibElementBoolInstVectorized,CmpLT,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches,
                                 const float * scale, const int32_t * offset, uint64_t flags);  

} // namespace dnn_lib
