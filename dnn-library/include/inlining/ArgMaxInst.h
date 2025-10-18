/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _ARG_MAX_INST_H_
#define _ARG_MAX_INST_H_

#include "Float16.h"
#include "LibCommon.h"
#include "LibTensor.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>
#include <type_traits>

namespace dnn_lib {

namespace inlining {

template <ElemKind outelK, ElemKind inelK>
INLINE_ATTR void fwdLibArgMaxInst(LibTensor* outT, LibTensor* inT, dim_t axis, bool keepDim, uint64_t flags,
                                  const uint32_t minionOffset = 0,
                                  [[maybe_unused]] const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  using inElemTy = typename elemKind2elemTy<inelK>::type;
  using outElemTy = typename elemKind2elemTy<outelK>::type;
  using midElemTy = typename std::conditional<((inelK == Float16Ty) || (inelK == FloatTy)), float, inElemTy>::type;

  assert(((inT->getElementType() == FloatTy) || (inT->getElementType() == Float16Ty) ||
          (inT->getElementType() == Int8QTy) || (inT->getElementType() == Int32QTy) ||
          (inT->getElementType() == Int32ITy) || (inT->getElementType() == Int64ITy)) &&
         "Not expected input Type");
  assert(((outT->getElementType() == Int64ITy) || 
          (outT->getElementType() == Int32ITy)) && "Not expected output Type");

  const dim_array_t& inpDims = inT->dims();

  auto outDims = inpDims;
  /* std::array<size_t, max_tensor_dimensions> outDims = inpDims; */
  outDims[axis] = 1;


  auto srcH = inT->getHandle<inElemTy>();
  auto destH = outT->getHandle<outElemTy>();
  
  for (dim_t idx0 = 0; idx0 < outDims[0]; idx0++) {
    for (dim_t idx1 = 0; idx1 < outDims[1]; idx1++) {
      for (dim_t idx2 = 0; idx2 < outDims[2]; idx2++) {
        for (dim_t idx3 = 0; idx3 < outDims[3]; idx3++) {
          for (dim_t idx4 = 0; idx4 < outDims[4]; idx4++) {
            for (dim_t idx5 = 0; idx5 < outDims[5]; idx5++) {

              // Init max value/index
              midElemTy maxVal = std::numeric_limits<midElemTy>::lowest();
              dim_t maxIdx = 0;

              //Iterate input axis dimension
              for(dim_t axisIdx = 0; axisIdx < inpDims[axis]; axisIdx++) {
                std::array<dim_t, max_tensor_dimensions> inpIdx = 
                  {idx0, idx1, idx2, idx3, idx4, idx5};
                
                inpIdx[axis] = axisIdx;

                midElemTy inpVal = static_cast<midElemTy>(0);

                if constexpr (inelK == Float16Ty) {
                  convertFp16ToFp32(static_cast<uint16_t>(srcH.at(inpIdx)), inpVal);
                } else {
                  inpVal = srcH.at(inpIdx);
                }

                if (inpVal > maxVal) {
                  maxVal = inpVal;
                  maxIdx = axisIdx;
                }
              }

              dim_array_t outIdx = {idx0, idx1, idx2, idx3, idx4, idx5};

              if (!keepDim) {
                for(size_t i = axis; i < (max_tensor_dimensions-1); i++) {
                  outIdx[i] = outIdx[i+1];
                }
                outIdx[5] = 1;
              }

              //Store maximum index.
              destH.at(outIdx) = static_cast<outElemTy>(maxIdx);
            }
          }
        }
      }
    }
  }
  outT->evict(DO_EVICTS);
}

} // namespace inlininig
} // namespace dnn_lib

#endif // _ARG_MAX_INST_H_
