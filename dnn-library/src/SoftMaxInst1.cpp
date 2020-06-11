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

#include "SoftMaxInst1.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elK>
void fwdLibSoftMaxInstThreaded1(LibTensor* outT, LibTensor* inT, uint64_t flags) {
  dnn_lib::inlining::fwdLibSoftMaxInstThreaded1<elK>(outT, inT, flags);
}

template <ElemKind elK>
void fwdLibSoftMaxInstVectorized1(LibTensor* outT, LibTensor* inT, uint64_t flags) {
  dnn_lib::inlining::fwdLibSoftMaxInstVectorized1<elK>(outT, inT, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSoftMaxInstThreaded1, LibTensor* outT, LibTensor* inT, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibSoftMaxInstVectorized1, LibTensor* outT, LibTensor* inT, uint64_t flags);
  
} // namespace dnn_lib
