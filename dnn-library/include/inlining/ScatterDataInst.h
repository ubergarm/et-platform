/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _SCATTER_DATA_INST_H_
#define _SCATTER_DATA_INST_H_

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

namespace ScatterDataImpl {
template <typename ElemTy>
void recurse(dim_array_t& inSliceIdx, dim_t inSliceP, LibTensor* slices, dim_array_t& outSliceIdx, dim_t outSliceP,
             LibTensor* out) {
  if (inSliceP == slices->ndims()) {
    // Copy the element from the input slice to the output
    // assert(outSliceP == out->ndims())
    auto OH = out->getHandle<ElemTy>();
    auto SH = slices->getHandle<ElemTy>();
    OH.at(outSliceIdx) = SH.at(inSliceIdx);
  } else {
    // Since we cannot assume data to be contiguous, set all indices for single element
    const dim_t* inSlicesDims = slices->dims().data();
    for (size_t i = 0; i < inSlicesDims[inSliceP]; ++i) {
      inSliceIdx[inSliceP] = i;
      outSliceIdx[outSliceP] = i;
      recurse<ElemTy>(inSliceIdx, inSliceP + 1, slices, outSliceIdx, outSliceP + 1, out);
    }
  }
}
} // namespace ScatterDataImpl

template <ElemKind elK>
INLINE_ATTR void fwdLibScatterDataInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                       [[maybe_unused]] uint64_t flags, const uint32_t minionOffset = 0,
                                       [[maybe_unused]] const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  if (get_minion_id() != minionOffset) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  /* outT --> dst  in1T--> src in2T--> slice*/

  const dim_t *indicesIndex = in1T->dims().data();

  dim_t sliceNumDim = in1T->ndims();

  dim_array_t slicesIt = in1T->dims();
  slicesIt[sliceNumDim - 1] = 1;
  dim_array_t sliceIdx;
  dim_array_t outIdx;

  auto IH = in1T->getHandle<uint64_t>();
  for (dim_t& k0 = sliceIdx[0] = 0; k0 < slicesIt[0]; ++k0) {
    for (dim_t& k1 = sliceIdx[1] = 0; k1 < slicesIt[1]; ++k1) {
      for (dim_t& k2 = sliceIdx[2] = 0; k2 < slicesIt[2]; ++k2) {
        for (dim_t& k3 = sliceIdx[3] = 0; k3 < slicesIt[3]; ++k3) {
          for (dim_t& k4 = sliceIdx[4] = 0; k4 < slicesIt[4]; ++k4) {
            for (dim_t& k5 = sliceIdx[5] = 0; k5 < slicesIt[5]; ++k5) {
              for (dim_t i = 0; i < indicesIndex[sliceNumDim - 1]; ++i) {
                // Set the first indicesIndex[sliceNumDim - 1] indices of the output slice
                sliceIdx[sliceNumDim - 1] = i;
                outIdx[i] = IH.at(sliceIdx);
              }
              ScatterDataImpl::recurse<srcType>(sliceIdx, sliceNumDim - 1, in2T, outIdx, indicesIndex[sliceNumDim - 1],
                                                outT);
            }
          }
        }
      }
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _SCATTER_DATA_INST_H_
