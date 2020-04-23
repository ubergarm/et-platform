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

#include "CopyInstTensorized.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibCopyInstTensorized(LibTensor* outT, LibTensor* inT, uint64_t flags,
                              const uint32_t minionOffset,
                              const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibCopyInstTensorized<srcType>(outT, inT, flags,
                                                       minionOffset,
                                                       assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibCopyInstTensorized, LibTensor* ouT, LibTensor* inT, uint64_t flags,
                 const uint32_t minionOffset, const uint32_t assignedMinions);
  
} // namespace dnn_lib
