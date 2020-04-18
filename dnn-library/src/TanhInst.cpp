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

#include "TanhInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibTanhInst(LibTensor* outT, LibTensor* inT) {

  dnn_lib::inlining::fwdLibTanhInst<srcType>(outT, inT);
}

template <typename srcType>
void fwdLibTanhInstThreaded(LibTensor* outT, LibTensor* inT, uint64_t flags) {

  dnn_lib::inlining::fwdLibTanhInstThreaded<srcType>(outT, inT, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibTanhInst, LibTensor* outT, LibTensor* inT);

GEN_INSTANCES_OP(template, fwdLibTanhInstThreaded, LibTensor* outT, LibTensor* inT, uint64_t flags);

} // namespace dnn_lib
