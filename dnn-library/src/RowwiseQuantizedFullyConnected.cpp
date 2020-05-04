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

#include "RowwiseQuantizedFullyConnected.h" // From include/inlining

namespace dnn_lib {

void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTy(LibTensor* outT,
      LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
      LibTensor* in5T) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTy(
                                     outT, in1T, in2T, in3T, in4T, in5T);
}

void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyThreaded(LibTensor* outT,
      LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
      LibTensor* in5T, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyThreaded(
                                     outT, in1T, in2T, in3T, in4T, in5T, flags);
}

void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyVectorized(LibTensor* outT,
      LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
      LibTensor* in5T, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyVectorized(
                                     outT, in1T, in2T, in3T, in4T, in5T, flags);
}

void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyAligned32Bytes(LibTensor* outT,
      LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
      LibTensor* in5T, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyAligned32Bytes(
                                     outT, in1T, in2T, in3T, in4T, in5T, flags);
}

} // namespace dnn_lib
