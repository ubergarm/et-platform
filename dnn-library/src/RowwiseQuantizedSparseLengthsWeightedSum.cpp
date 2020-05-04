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

void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
    LibTensor* in4T, LibTensor* in5T, LibTensor* in6T) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
                                      outT, in1T, in2T, in3T, in4T, in5T, in6T);
}

void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
             LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
             LibTensor* in4T, LibTensor* in5T, LibTensor* in6T, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
                                outT, in1T, in2T, in3T, in4T, in5T, in6T, flags);
}

template<bool Int8Src, bool Float16Dst>
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized(
       LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
       LibTensor* in4T, LibTensor* in5T, LibTensor* in6T, uint64_t flags,
       const uint32_t minionOffset, const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized<Int8Src, Float16Dst>(
                outT, in1T, in2T, in3T, in4T, in5T, in6T, flags,
                minionOffset, assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_RQSLWS_V(template, fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized,
          LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
          LibTensor* in4T, LibTensor* in5T, LibTensor* in6T, uint64_t flags,
	    		const uint32_t minionOffset, const uint32_t assignedMinions);

} // namespace dnn_lib
