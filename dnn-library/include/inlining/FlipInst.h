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

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "Float16.h"
#include "utils.h"
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
inline void fwdLibFlipInst(LibTensor* outT, LibTensor* inT, unsigned int axis,
                           uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  using ElemTy = typename elemKind2elemTy<elK>::type;

  auto srcH = inT->getHandle<ElemTy>();
  auto destH = outT->getHandle<ElemTy>();
  
  loopAxis(srcH, destH, inT->dims(), axis);

}

}//namespace inlining

} //namespace dnn_lib

#endif  // _FLIP_INST_H_
