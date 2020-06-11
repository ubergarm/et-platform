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

#include "SigmoidInst.h" // From include/inlining path

namespace dnn_lib {

template <ElemKind elK>
void fwdLibSigmoidInstThreaded(LibTensor* outT, LibTensor* inT, uint64_t flags) {

  dnn_lib::inlining::fwdLibSigmoidInstThreaded<elK>(outT, inT, flags);
}

template <ElemKind elK>
void fwdLibSigmoidInst(LibTensor* outT, LibTensor* inT) {

  dnn_lib::inlining::fwdLibSigmoidInst<elK>(outT, inT);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSigmoidInst, LibTensor* outT, LibTensor* inT);
GEN_INSTANCES_OP(template, fwdLibSigmoidInstThreaded, LibTensor* outT, LibTensor* inT, uint64_t flags);

} // namespace dnn_lib
