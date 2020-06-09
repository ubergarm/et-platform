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

#include "SyncopyInstTensorized.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibSyncopyInstTensorized(LibTensor* outT, LibTensor* inT, unsigned int off,
                                 const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibSyncopyInstTensorized<srcType>(outT, inT, off, assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSyncopyInstTensorized, LibTensor* outT, LibTensor* inT,
                                  unsigned int off, const uint32_t assignedMinions);

} // namespace dnn_lib
