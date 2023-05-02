/*-------------------------------------------------------------------------
 * Copyright (C) 2023, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _TRILU_INST_H_
#define _TRILU_INST_H_

#include "Float16.h"
#include "LibTensor.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
INLINE_ATTR void fwdLibTriluInst([[maybe_unused]] LibTensor* outT, [[maybe_unused]] LibTensor* inTData,
                                 [[maybe_unused]] LibTensor* inTDiags, [[maybe_unused]] bool upper,
                                 [[maybe_unused]] uint64_t flags, [[maybe_unused]] const uint32_t minionOffset = 0,
                                 [[maybe_unused]] const uint32_t assignedMinions = 0) {
  // FIXME: TODO: implement
}

} // namespace inlining

} // namespace dnn_lib

#endif //  _TRILU_INST_H_
