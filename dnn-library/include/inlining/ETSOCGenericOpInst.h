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

#include "DenoiseMasks.h"
#include "LibTensor.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

enum class Operation { FFT = 0, IFFT = 1, FFT_FILTER_IFFT = 2, NOISE_FILTER_1 = 3, LAST };
static constexpr const char* Op2String[] = {"FFT", "IFFT", "FFT_FILTER_IFFT", "NOISE_FILTER_1", "LAST"};

INLINE_ATTR void fft(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset,
                     const uint32_t assignedMinions, uint32_t activeMinions, uint32_t minionId) {

  //  FIXME: just minon 0 does some work at the moment.
  if (minionId != 0) {
    return;
  }
  (void)flags;
  (void)minionOffset;
  (void)activeMinions;
  (void)assignedMinions;

  et_printf("%s(%d) [%d]\n", __func__, __LINE__, minionId);

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
}

INLINE_ATTR void freqDomainNoiseFilter(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset,
                                       const uint32_t assignedMinions, uint32_t activeMinions, uint32_t minionId) {
  //  FIXME: just minon 0 does some work at the moment.
  if (minionId != 0) {
    return;
  }

  et_printf("%s(%d) [%d]\n", __func__, __LINE__, minionId);

  (void)flags;
  (void)minionOffset;
  (void)activeMinions;
  (void)assignedMinions;

  et_assert(inT->dims()[0] == 2);

  auto w = inT->dims()[1];
  auto h = inT->dims()[2];
  auto inH = inT->getHandle<float>();
  auto outH = outT->getHandle<float>();
  // elementwise product of real and imaginary planes with the mask.
  // it is just  a PoC. it can probably be just fused with fft.
  // (Evey real or imaginary output emited or not based on mask
  // also SIMD mask's can be used ot skip when fft is vectorized.
  for (dim_t n = 0; n < w; n++) {
    for (dim_t m = 0; m < h; m++) {
      std::array<dim_t, 3> posReal = {0, m, n};
      std::array<dim_t, 3> posImg = {1, m, n};
      auto mask = denoiseMask[m * w + n];

      outH.at(posReal) = inH.at(posReal) * mask;
      outH.at(posImg) = inH.at(posImg) * mask;
    }
  }
}

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

  switch (Operation(op)) {
  case Operation::FFT:
    return fft(outT, inT, flags, minionOffset, assignedMinions, activeMinions, minionId);
  case Operation::NOISE_FILTER_1:
    return freqDomainNoiseFilter(outT, inT, flags, minionOffset, assignedMinions, activeMinions, minionId);
  default:
    et_assert("unsupported operation");
  }

  return;
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ETSOCGENERICOP_INST_H_
