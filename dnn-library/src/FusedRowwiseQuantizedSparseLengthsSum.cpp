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

#include "FusedRowwiseQuantizedSparseLengthsSum.h" // From include/inlining

namespace dnn_lib {

template<typename DstType>
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstFloatTyThreaded(
             LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
             uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsSumInstFloatTyThreaded<DstType>(
             outT, in1T, in2T, in3T, flags,
             minionOffset, assignedMinions);
}

template<typename DstType>
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstFloatTyVectorized(
             LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
             uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsSumInstFloatTyVectorized<DstType>(
             outT, in1T, in2T, in3T, flags,
             minionOffset, assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_1TYPEFP(template, fwdLibFusedRowwiseQuantizedSparseLengthsSumInstFloatTyThreaded,
                      LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                      uint64_t flags, const uint32_t minionOffset, const uint32_t numShires);
GEN_INSTANCES_1TYPEFP(template, fwdLibFusedRowwiseQuantizedSparseLengthsSumInstFloatTyVectorized,
                      LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                      uint64_t flags, const uint32_t minionOffset, const uint32_t numShires);

} // namespace dnn_lib
