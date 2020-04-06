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

#include "ElementInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType, typename opType>
void fwdLibElementInst(void *dstT, void *dstDims, void *dstPitches,
                                void *srcT1, void *srcDims, void *src1Pitches,
                                unsigned int srcDimNum, void *srcT2,
                                void *src2Pitches, const float *scale,
                                const int32_t *offset) {

  dnn_lib::inlining::fwdLibElementInst<srcType, opType>(dstT, dstDims, dstPitches,
                                srcT1, srcDims, src1Pitches,
                                srcDimNum, srcT2,
                                src2Pitches, scale,
                                offset);

}

template <typename src1Type, typename src2Type, typename dstType, typename opType>
void fwdLibElementInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *src1Pitches, unsigned int srcDimNum, void *srcT2, void *src2Pitches, const float *scale,
    const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementInstThreaded<src1Type, src2Type, dstType, opType>(
    dstT, dstDims, dstPitches, srcT1, srcDims,
    src1Pitches, srcDimNum, srcT2, src2Pitches, scale,
    offset, flags);
}

template <typename src1Type, typename src2Type, typename dstType, typename opType>
void fwdLibElementInstVectorized(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *src1Pitches, unsigned int srcDimNum, void *srcT2, void *src2Pitches,
    const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementInstVectorized<src1Type, src2Type, dstType, opType>(
    dstT, dstDims, dstPitches, srcT1, srcDims,
    src1Pitches, srcDimNum, srcT2, src2Pitches,
    scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Add,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Sub,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Div,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Mul,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Max,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Min,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Pow,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Add,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches, const float * scale,
                        const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Sub,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches, const float * scale,
                        const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Div,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches, const float * scale,
                        const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Mul,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches, const float * scale,
                        const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Max,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches, const float * scale,
                        const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Min,void *dstT, void *dstDims,
                        void *dstPitches, void *srcT1,
                        void *srcDims, void *src1Pitches,
                        unsigned int srcDimNum, void *srcT2,
                        void *src2Pitches, const float * scale,
                        const  int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Pow,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Add,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Sub,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Div,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Mul,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Max,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Min,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Pow,void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *srcT2,
                                 void *src2Pitches, const float * scale,
                                 const int32_t * offset, uint64_t flags);
}
