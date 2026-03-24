/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _ONE_HOT_INST_H_
#define _ONE_HOT_INST_H_

#include "Addresser.h"
#include "Float16.h"
#include "LibTensor.h"
#include "etsoc/common/utils.h"
#include "utils.h"
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

/**


 * @param[out] outT LibTensor pointer to the output tensor.
 * @param[in] in1T LibTensor pointer to the indices tensor.
 * @param[in] int2T LibTensor pointer to the depth tensor.
 * @param[in] int3T LibTensor pointer to the values tensor.
 * @param[in] axis integer that contains axis value.
 * @param[in] flags Controls the active shires and the type of evict that
 * should be done at the end of the function.
 */

template <typename ElemTy, typename IndexElem>
void process_vector(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, dim_array_t& outputCoord,
                    dim_array_t& indexCoord, uint64_t curr_height, uint64_t& depth, uint64_t& axis, uint64_t reached) {

  // Tensor handles:
  auto indicesH = in1T->getHandle<IndexElem>();
  auto valuesH = in2T->getHandle<ElemTy>();
  auto outputH = outT->getHandle<ElemTy>();
  auto height = outT->ndims();
  if (height == curr_height) {
    outputH.at(outputCoord) = (outputCoord[axis] == (dim_t)indicesH.at(indexCoord) ? valuesH.raw(1) : valuesH.raw(0));
  } else if (curr_height == axis) {
    reached = 1;
    for (dim_t i = 0; i < (dim_t)depth; ++i) {
      outputCoord[curr_height] = i;
      process_vector<ElemTy, IndexElem>(outT, in1T, in2T, outputCoord, indexCoord, curr_height + 1, depth, axis,
                                        reached);
    }
  } else {
    for (dim_t i = 0; i < outT->dims()[curr_height]; ++i) {
      outputCoord[curr_height] = i;
      indexCoord[curr_height - reached] = i;
      process_vector<ElemTy, IndexElem>(outT, in1T, in2T, outputCoord, indexCoord, curr_height + 1, depth, axis,
                                        reached);
    }
  }
}

template <ElemKind outelK, ElemKind inelK>
INLINE_ATTR void fwdLibOneHotInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t axis_int, uint64_t depth,
                                  uint64_t flags, [[maybe_unused]] const uint32_t minionOffset = 0,
                                  [[maybe_unused]] const uint32_t assignedMinions = 0) {

  using ElemTy = typename elemKind2elemTy<outelK>::type;
  using IndexElem = typename elemKind2elemTy<inelK>::type;

  if (get_minion_id() != minionOffset)
    return;

  et_assert(in2T->getElementType() == outT->getElementType()); // values and output must have the same tipus

  uint64_t axis = (axis_int < 0 ? (int64_t)in1T->ndims() + axis_int + 1 : axis_int);

  uint64_t curr_height = 0;
  dim_array_t outputCoord;
  dim_array_t indexCoord;

  process_vector<ElemTy, IndexElem>(outT, in1T, in2T, outputCoord, indexCoord, curr_height, depth, axis, 0);

  outT->evict(DO_EVICTS);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ONE_HOT_INST_H_