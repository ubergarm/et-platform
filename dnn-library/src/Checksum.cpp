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

#include "Checksum.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibChecksum(void *src, void *srcDims, void *srcPitches,
                             unsigned int srcDimNum, const float *scale,
                             const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibChecksum<srcType>(src, srcDims, srcPitches,
                             srcDimNum, scale,
                             offset, flags);
}

void fwdLibFlushL3(uint32_t numShires) {
  dnn_lib::inlining::fwdLibFlushL3(numShires);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibChecksum, void *src, void *srcDims, void *srcPitches,
                                  unsigned int srcDimNum,
                                  const float *scale, const int32_t *offset,
                                  uint64_t flags);
}
