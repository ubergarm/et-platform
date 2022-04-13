/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _ETSOCGENERICOP_INST_H_
#define _ETSOCGENERICOP_INST_H_

#include "LibTensor.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>
#include <string_view>

namespace dnn_lib {

namespace inlining {

enum class Operation { FFT = 0, IFFT = 1, FFT_FILTER_IFFT = 2, NOISE_FILTER_1 = 3, LAST };
static constexpr const char* Op2String[] = {"FFT", "IFFT", "FFT_FILTER_IFFT", "NOISE_FILTER_1", "LAST"};

template <ElemKind elK>
INLINE_ATTR void fwdLibETSOCGenericOpInst(LibTensor* outT, LibTensor* inT, uint32_t op, uint64_t flags,
                                          const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  (void)flags;

  static_assert(elK == FloatTy);

  et_printf("%s(%d) [%d] called with op: %s \n", __func__, __LINE__, get_minion_id(), Op2String[op]);

  auto minionId = get_minion_id();
  // Rebase minion ID.
  minionId -= minionOffset;

  // Get number of Minions assigned to this Node.
  uint64_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;

  // If Minion is outside the group assigned to this Node get out.
  if (minionId >= activeMinions) {
    return;
  }

  //  FIXME: just minon 0 does some work at the moment.
  if (minionId != 0) {
    return;
  }

  et_printf("%s(%d) [%d]\n", __func__, __LINE__, get_minion_id());

  // just copy input over real and imaginary planes.
  auto inH = inT->getHandle<float>();
  auto outH = outT->getHandle<float>();
  auto outIt = outH.getIterator(0);

  for (auto inIt = inH.getIterator(0); inIt != inH.end(); ++inIt, ++outIt) {
    *outIt = *inIt;
  }

  for (auto inIt = inH.getIterator(0); inIt != inH.end(); ++inIt, ++outIt) {
    *outIt = *inIt;
  }

  return;
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ETSOCGENERICOP_INST_H_
