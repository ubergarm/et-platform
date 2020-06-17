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

#include "SparseLengthsWeightedSumInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elKind, ElemKind idxKind>
void fwdLibSparseLengthsWeightedSumInst(LibTensor* outT, LibTensor* in1T,
                                        LibTensor* in2T, LibTensor* in3T,
                                        LibTensor* in4T,
					uint64_t flags) {

  dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInst<elKind, idxKind>(outT, in1T,
                                                                 in2T, in3T,
                                                                 in4T, flags);
}

template <ElemKind elKind, ElemKind idxKind>     
void fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* outT, LibTensor* in1T,
                                                LibTensor* in2T, LibTensor* in3T,
                                                LibTensor* in4T,
                                                uint64_t flags) {
                     
  dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInstThreaded<elKind, idxKind>(outT, in1T,
                                                                         in2T, in3T,
                                                                         in4T, flags);
}

} // namespace dnn_lib
