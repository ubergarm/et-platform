/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _PROFILE_INST_H_
#define _PROFILE_INST_H_

#include "Float16.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

#include <etsoc/isa/barriers.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include "LibTensor.h"
#include "utils.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Dumps profile information of a tensor to another tensor
 *
 * This function traverses a buffer and writes max and min values to
 * another one.
 *
 * @tparam srcType The type of the elements in the matrices.
 * @param[in] src Pointer to origin buffer to profile
 * @param[in] dst Pointer to destination buffer to write to
 * @param[in] off offset in CL where to write the results
 */
template <ElemKind elK>
INLINE_ATTR void fwdLibProfileInst(LibTensor* outT, LibTensor* inT, unsigned int off, uint64_t flags,
                                   const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  assert((minionOffset + activeMinions <= MIN_PER_SHIRE * activeShires(flags)) && "Minion ID overflow");

  // Total number of elements in the tensor (not accounting padding elements)
  auto numElems = inT->size();

  // Need a power of two number of active minions. Make sure if there's minionOffset that the active minions
  // block is aligned to the power of two number of active minions. Also, don't activate more minions than the
  // number of elements to process.
  size_t p2_minions = 0;
  size_t minions = 1;
  while ((minions < activeMinions) and ((minionOffset == 0) or (minions < minionOffset)) and (minions < numElems)) {
    p2_minions++;
    minions *= 2;
  }
  activeMinions = minions;
  assert((minionOffset % activeMinions == 0));
  if (minionId >= activeMinions) {
    return;
  }

  // Distribute elements per minion.
  auto elemsMinion = numElems / activeMinions;
  auto elemsReminder = numElems % activeMinions;
  auto firstMinionDoingOneExtra = activeMinions - elemsReminder;
  size_t initialNonPaddingOffset = elemsMinion * minionId;
  if (minionId >= firstMinionDoingOneExtra) {
    elemsMinion++;
    initialNonPaddingOffset += (minionId - firstMinionDoingOneExtra);
  }

  // Initialize the coordinates of the first element to process.
  dim_array_t coord = {0};
  auto inputPitchNoPadding = inT->stridesNoPadding();
  getCoordinates(coord, initialNonPaddingOffset, inT->ndims(), inputPitchNoPadding.data());

  // Get the actual initialAddr in the input tensor.
  size_t offsetIn = 0;
  const dim_t* srcPitch = inT->strides().data();
  for (size_t j = 0; j < inT->ndims(); j++) {
    offsetIn += srcPitch[j] * coord[j];
  }

  // Traverse the segment of assigned elements and keep track of min and max values
  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::lowest();
  auto src = inT->getRawDataPointer<srcType>();
  const Addresser<elK> tInput(src, inT->getScale(), inT->getOffset());
  for (size_t j = 0; j < elemsMinion; j++) {
    float val = float(tInput[offsetIn]);
    min = (val < min) ? val : min;
    max = (val > max) ? val : max;
    advanceOffsetAndCoordinates(coord, inT->dims(), offsetIn, inT->ndims(), inT->strides());
  }

  // Reduce across all active minions
  for (size_t i = 0; i < p2_minions; i++) {
    min = tensor_reduce_float(min, 0x3, 1, i, 0x3);
    max = tensor_reduce_float(max, 0x2, 1, i, 0x3);
  }

  // Write result
  auto dst = outT->getRawDataPointer<float>();
  if (minionId == 0) {
    dst[16 * off] = min;
    dst[16 * off + 1] = max;
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _COPY_INST_H_
