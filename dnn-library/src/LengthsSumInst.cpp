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

#include "LengthsSumInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibLengthsSumInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                          unsigned int pLengthsSize) {

  dnn_lib::inlining::fwdLibLengthsSumInst<srcType>(outT, in1T, in2T, pLengthsSize);
}

template <typename srcType>
void fwdLibLengthsSumInstThreaded(LibTensor* outT, LibTensor* in1T,
                                  LibTensor* in2T, uint64_t flags) {

  dnn_lib::inlining::fwdLibLengthsSumInstThreaded<srcType>(outT, in1T, in2T, flags);

}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibLengthsSumInst, LibTensor* outT, LibTensor* in1T,
                 LibTensor* in2T, unsigned int pLengthsSize);

GEN_INSTANCES_OP(template, fwdLibLengthsSumInstThreaded, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, uint64_t flags);
} // namespace dnn_lib
