/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
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
