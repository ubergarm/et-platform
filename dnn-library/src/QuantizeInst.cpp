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

#include "QuantizeInst.h" // From include/inlining

namespace dnn_lib {

template <typename dstType>
void fwdLibQuantizeInst(void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, float scale,
                                 int32_t offset) {

  dnn_lib::inlining::fwdLibQuantizeInst<dstType>(dstT, dstDims, dstPitches,
                                 srcT, srcDims, srcPitches,
                                 srcDimNum, scale,
                                 offset);
}

template <typename dstType>
void fwdLibQuantizeInstThreaded(void *dstT, void *dstDims,
                                         void *dstPitches, void *srcT,
                                         void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float scale,
                                         int32_t offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibQuantizeInstThreaded<dstType>(dstT, dstDims,
                                         dstPitches, srcT,
                                         srcDims, srcPitches,
                                         srcDimNum, scale,
                                         offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_QUANT(template, fwdLibQuantizeInst, void *dstT, void *dstDims, void *dstPitches,
                              void *srcT, void *srcDims, void *srcPitches, unsigned int srcDimNum,
                              float scale,  int32_t offset);

GEN_INSTANCES_QUANT(template, fwdLibQuantizeInstThreaded, void *dstT, void *dstDims, void *dstPitches,
                              void *srcT, void *srcDims, void *srcPitches, unsigned int srcDimNum,
                              float scale,  int32_t offset, uint64_t flags);
} // namespace dnn_lib
