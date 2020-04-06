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

#include "ElementImmLogic.h" // From include/inlining

namespace dnn_lib {

template <typename srcType, typename opType>
void fwdLibElementImmLogic(void *dstT, void *dstDims,
                                     void *dstPitches, void *srcT1,
                                     void *srcDims, void *srcPitches,
                                     unsigned int srcDimNum, void *imm,
                                     const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibElementImmLogic<srcType, opType>(dstT, dstDims,
                                     dstPitches, srcT1,
                                     srcDims, srcPitches,
                                     srcDimNum, imm,
                                     scale, offset);
}

#include "GenInstances.h"

GEN_INSTANCES_INSTANCES_LOGIC(template, fwdLibElementImmLogic, And, void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *imm, 
                                 const float * scale, const int32_t * offset);

GEN_INSTANCES_INSTANCES_LOGIC(template, fwdLibElementImmLogic, Or, void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *imm, 
                                 const float * scale, const int32_t * offset);

GEN_INSTANCES_INSTANCES_LOGIC(template, fwdLibElementImmLogic, Xor, void *dstT, void *dstDims,
                                 void *dstPitches, void *srcT1,
                                 void *srcDims, void *src1Pitches,
                                 unsigned int srcDimNum, void *imm, 
                                 const float * scale, const int32_t * offset);

} // namespace dnn_lib
