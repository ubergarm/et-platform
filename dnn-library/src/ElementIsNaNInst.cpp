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

#include "ElementIsNaNInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibElementIsNaNInst(LibTensor* outT, LibTensor *inT) {

  dnn_lib::inlining::fwdLibElementIsNaNInst<srcType>(outT, inT);
}

template <typename srcType>
void fwdLibElementIsNaNInstThreaded(LibTensor* outT, LibTensor* inT,
                                    uint64_t flags) {

  dnn_lib::inlining::fwdLibElementIsNaNInstThreaded<srcType>(outT, inT, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibElementIsNaNInst, LibTensor* outT, LibTensor* inT);
GEN_INSTANCES_OP(template, fwdLibElementIsNaNInstThreaded, LibTensor* outT, LibTensor* inT, uint64_t flags);

} // namespace dnn_lib
