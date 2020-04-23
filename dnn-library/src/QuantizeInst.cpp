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

#include "QuantizeInst.h" // From include/inlining

namespace dnn_lib {

template <typename dstType>
void fwdLibQuantizeInst(LibTensor* outT, LibTensor* inT) {

  dnn_lib::inlining::fwdLibQuantizeInst<dstType>(outT, inT);
}

template <typename dstType>
void fwdLibQuantizeInstThreaded(LibTensor* outT, LibTensor* inT, uint64_t flags) {

  dnn_lib::inlining::fwdLibQuantizeInstThreaded<dstType>(outT, inT, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_QUANT(template, fwdLibQuantizeInst, LibTensor* outT, LibTensor* inT);

GEN_INSTANCES_QUANT(template, fwdLibQuantizeInstThreaded,  LibTensor* outT, LibTensor* inT, uint64_t flags);
} // namespace dnn_lib
