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

#include "RescaleQuantizedInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibRescaleQuantizedInst(void *dstT, void *dstDims,
                                         void *dstPitches, void *srcT,
                                         void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float srcScale,
                                         int32_t srcOffset, float dstScale,
                                         int32_t dstOffset) {

  dnn_lib::inlining::fwdLibRescaleQuantizedInst<srcType>(dstT, dstDims,
                                         dstPitches, srcT,
                                         srcDims, srcPitches,
                                         srcDimNum, srcScale,
                                         srcOffset, dstScale,
                                         dstOffset);
}

template <typename srcType>
void fwdLibRescaleQuantizedInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, float srcScale, int32_t srcOffset,
    float dstScale, int32_t dstOffset, uint64_t flags) {

  dnn_lib::inlining::fwdLibRescaleQuantizedInstThreaded<srcType>(
    dstT, dstDims, dstPitches, srcT, srcDims,
    srcPitches, srcDimNum, srcScale, srcOffset,
    dstScale, dstOffset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_QUANT(template, fwdLibRescaleQuantizedInst, void *dstT, void *dstDims, void *dstPitches,
                                      void *srcT, void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, float srcScale,
                                      int32_t srcOffset, float dstScale, int32_t dstOffset);

GEN_INSTANCES_QUANT(template, fwdLibRescaleQuantizedInstThreaded, void *dstT, void *dstDims, void *dstPitches,
                                      void *srcT, void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, float srcScale, int32_t srcOffset,
                                      float dstScale, int32_t dstOffset, uint64_t flags);
} // namespace dnn_lib
