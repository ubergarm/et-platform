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

#include "ArgMaxInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibArgMaxInst( void *src, void *srcDims, void *srcPitches, float srcScale, int32_t srcOffset,
                                void *dst, void *dstDims, void *dstPitches,
                                size_t axis, bool keepDim){

  dnn_lib::inlining::fwdLibArgMaxInst<srcType>(src, srcDims, srcPitches, srcScale, srcOffset,
                                dst, dstDims, dstPitches,
                                axis, keepDim);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibArgMaxInst,
                 void *src, void *srcDims, void *srcPitches, float srcScale, int32_t srcOffset,
                 void *dst, void *dstDims, void *dstPitches,
                 size_t axis, bool keepDim);
} // dnn_lib
