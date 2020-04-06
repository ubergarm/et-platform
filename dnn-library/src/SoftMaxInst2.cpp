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

#include "SoftMaxInst2.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibSoftMaxInst2(void *dstT, void *srcT, void *srcTDims,
                                 void *srcTPitches, const float *scale,
                                 const int32_t *offset) {

  dnn_lib::inlining::fwdLibSoftMaxInst2<srcType>(dstT, srcT, srcTDims,
                                 srcTPitches, scale,
                                 offset);
}

template <typename srcType>
void fwdLibSoftMaxInstThreaded2 (void *dstT, void *srcT, void *srcTDims,
                                          void *srcTPitches, const float *scale,
                                          const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibSoftMaxInstThreaded2<srcType>(dstT, srcT, srcTDims,
                                          srcTPitches, scale,
                                          offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSoftMaxInst2, void *dstT, void *srcT, void *srcTDims,
                          void *srcTPitches, const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibSoftMaxInstThreaded2, void *dstT, void *srcT, void *srcTDims,
                          void *srcTPitches, const float *scale, const int32_t *offset, uint64_t flags);


} // namespace dnn_lib
