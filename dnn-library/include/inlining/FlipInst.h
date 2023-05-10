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

#ifndef FLIP_INST_H_
#define FLIP_INST_H_

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
INLINE_ATTR void fwdLibFlipInst(LibTensor* outT, LibTensor* inT, dim_t axis, [[maybe_unused]] uint64_t flags,
                                const uint32_t minionOffset = 0, [[maybe_unused]] const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  using ElemTy = typename elemKind2elemTy<elK>::type;

  auto srcH = inT->getHandle<ElemTy>();
  auto destH = outT->getHandle<ElemTy>();
  
  loopAxis(srcH, destH, inT->dims(), axis);
}

}//namespace inlining

} //namespace dnn_lib

#endif  // _FLIP_INST_H_
