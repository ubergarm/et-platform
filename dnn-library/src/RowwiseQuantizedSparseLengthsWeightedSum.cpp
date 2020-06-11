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

#include "RowwiseQuantizedSparseLengthsWeightedSum.h" // From include/inlining

namespace dnn_lib {
template <ElemKind dstElK, ElemKind srcElK>
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst(
    LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
    LibTensor* in4T, LibTensor* in5T, LibTensor* in6T) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<dstElK, srcElK>(
                                      outT, in1T, in2T, in3T, in4T, in5T, in6T);
}

  template <ElemKind dstElK, ElemKind srcElK>
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstThreaded(
             LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
             LibTensor* in4T, LibTensor* in5T, LibTensor* in6T, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstThreaded<dstElK, srcElK>(
                                outT, in1T, in2T, in3T, in4T, in5T, in6T, flags);
}

template <ElemKind dstElK, ElemKind srcElK>
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized(
       LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
       LibTensor* in4T, LibTensor* in5T, LibTensor* in6T, uint64_t flags,
       const uint32_t minionOffset, const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<dstElK, srcElK>(
                outT, in1T, in2T, in3T, in4T, in5T, in6T, flags,
                minionOffset, assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_RQSLWS_V(template, fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized,
          LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
          LibTensor* in4T, LibTensor* in5T, LibTensor* in6T, uint64_t flags,
	    		const uint32_t minionOffset, const uint32_t assignedMinions);

} // namespace dnn_lib
