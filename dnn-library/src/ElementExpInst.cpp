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

#include "ElementExpInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibElementExpInst(void *dstT, void *dstDims, void *dstPitches,
                                   void *srcT, void *srcDims, void *srcPitches,
                                   unsigned int srcDimNum, const float *scale,
                                   const int32_t *offset) {

  dnn_lib::inlining::fwdLibElementExpInst<srcType>(dstT, dstDims, dstPitches,
                                   srcT, srcDims, srcPitches,
                                   srcDimNum, scale,
                                   offset);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibElementExpInst,void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT1, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, const float * scale,
                                 const int32_t * offset);
} // namespace dnn_lib
