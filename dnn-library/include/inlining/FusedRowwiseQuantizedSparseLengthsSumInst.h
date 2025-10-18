/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _FUSED_ROWWISE_QUANTIZED_SPARSE_LENGTHS_SUM_H_
#define _FUSED_ROWWISE_QUANTIZED_SPARSE_LENGTHS_SUM_H_

#include "Float16.h"
#include "FusedRowwiseQuantizedSparseLengthsWeightedSumInst.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
inline __attribute__((always_inline))
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst(
        LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
        uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  dnn_lib::inlining::fusedRowwiseQuantizedSparseLengthsWeightedSumInstVectorizedImpl<elK, false>
    (outT, in1T, nullptr, in2T, in3T, flags, minionOffset, assignedMinions);
}



  
} // namespace inlining

} // namespace dnn_lib

#endif // _FUSED_ROWWISE_QUANTIZED_SPARSE_LENGTHS_SUM_H_
