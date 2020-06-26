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

#ifndef _FUSED_ROWWISE_QUANTIZED_SPARSE_LENGTHS_SUM_H_
#define _FUSED_ROWWISE_QUANTIZED_SPARSE_LENGTHS_SUM_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "Float16.h"
#include "Writer.h" // From include/internal path
#include "Addresser.h" // From include/internal path
#include "Converter.h" // From include/internal path
#include "Operator.h" // From include/internal path
#include "utils.h" // From include/internal path
#include "FusedRowwiseQuantizedSparseLengthsWeightedSumInst.h" // From include/inlining path

namespace dnn_lib {

namespace inlining {
  
template <ElemKind elK>
inline __attribute__((always_inline))
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstThreaded(
        LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
        uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  LibTensor* inW = nullptr;
  dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstThreaded <elK> (
    outT, in1T, inW, in2T, in3T, flags, minionOffset, assignedMinions);
}

template <ElemKind elK>
inline __attribute__((always_inline))
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstVectorized(
        LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
        uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  dnn_lib::inlining::fusedRowwiseQuantizedSparseLengthsWeightedSumInstVectorizedImpl<elK, false>
    (outT, nullptr, in1T, nullptr, in2T, in3T, flags, minionOffset, assignedMinions);
}

template <ElemKind elK>
inline __attribute__((always_inline))
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst(
        LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
        uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  // there's no single node version => forward call to threaded implementation
  dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsSumInstThreaded<elK>(outT, in1T, in2T, in3T,
                                                                                  flags, minionOffset, assignedMinions);
}



  
} // namespace inlining

} // namespace dnn_lib

#endif // _FUSED_ROWWISE_QUANTIZED_SPARSE_LENGTHS_SUM_H_
