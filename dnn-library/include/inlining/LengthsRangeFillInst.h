/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _LENGTHS_RANGE_FILL_H_
#define _LENGTHS_RANGE_FILL_H_

#include "LibCommon.h"
#include "LibTensor.h"
#include "LibTypes.h"
#include "utils.h"
#include <assert.h>

namespace dnn_lib {

namespace inlining {

/**
 * @brief Convert a length vector to a range sequence. 
 *
 * For example, input=[4,3,1], the output would be [0,1,2,3,0,1,2,0].
 *
 * Currently It only solves Int32ITy ElemKind following InstGen.cpp
 * Interpreter.cpp and isOpSupported at ETSOC.cpp specification.
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected data.
 * @param[in] inT LibTensor input. It keeps the inputs to being handle.
 * @param[flags] flags Gives the information of the Active Shires and the
 *  type of evict required.
 */
template <ElemKind elKind>
inline void fwdLibLengthsRangeFillInst(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset = 0,
                                       [[maybe_unused]] const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(inT->getElementType() == outT->getElementType());
  assert(inT->getElementType() == Int32ITy);

  using elkType = typename elemKind2elemTy<elKind>::type;

  auto inH = inT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();
  
  std::array<size_t, 1> curIdx = {0};

  for(size_t i = 0; i < inT->dims()[0]; i++) {
    std::array<size_t, 1> atIdx = {i};
    for (int32_t j = 0; j < inH.at(atIdx); j++) {
      outH.at(curIdx) = j;
      curIdx[0]++;
    }
  }
  
  outT->evict(DO_EVICTS);
}
}
}
#endif
