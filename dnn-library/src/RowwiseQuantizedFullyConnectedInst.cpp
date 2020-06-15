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

#include "RowwiseQuantizedFullyConnectedInst.h" // From include/inlining

namespace dnn_lib {

void fwdLibRowwiseQuantizedFullyConnectedInst(LibTensor* outT,
      LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
      LibTensor* in5T) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInst(
                                     outT, in1T, in2T, in3T, in4T, in5T);
}

void fwdLibRowwiseQuantizedFullyConnectedInstThreaded(LibTensor* outT,
      LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
      LibTensor* in5T, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstThreaded(
                                     outT, in1T, in2T, in3T, in4T, in5T, flags);
}

void fwdLibRowwiseQuantizedFullyConnectedInstVectorized(LibTensor* outT,
      LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
      LibTensor* in5T, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstVectorized(
                                     outT, in1T, in2T, in3T, in4T, in5T, flags);
}

void fwdLibRowwiseQuantizedFullyConnectedInstAligned32Bytes(LibTensor* outT,
      LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
      LibTensor* in5T, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstAligned32Bytes(
                                     outT, in1T, in2T, in3T, in4T, in5T, flags);
}

} // namespace dnn_lib
