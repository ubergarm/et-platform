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

#include "IntLookupTableInst.h" // From include/inlining

namespace dnn_lib {

void fwdLibIntLookupTableInstInt8QTy(LibTensor* outT, LibTensor* in1T,
                                     LibTensor* in2T) {

  dnn_lib::inlining::fwdLibIntLookupTableInstInt8QTy(outT, in1T, in2T);

}

void fwdLibIntLookupTableInstInt8QTyThreaded(LibTensor* outT, LibTensor* in1T,
                                             LibTensor* in2T, uint64_t flags) {
    
  dnn_lib::inlining::fwdLibIntLookupTableInstInt8QTyThreaded(outT, in1T, in2T, flags);

}

} // namespace dnn_lib
