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

#include "ModuloInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibModuloInst(LibTensor* outT, LibTensor* inT, long long divisor,
                      bool signFollowDivisor) {

  dnn_lib::inlining::fwdLibModuloInst<srcType>(outT, inT, divisor,
                                               signFollowDivisor);
}

template <typename srcType>
void fwdLibModuloInstThreaded(LibTensor* outT, LibTensor* inT, long long divisor,
                              bool signFollowDivisor, uint64_t flags) {

  dnn_lib::inlining::fwdLibModuloInstThreaded<srcType>(outT, inT, divisor,
                                                       signFollowDivisor, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_INTONLY_OP(template, fwdLibModuloInst, LibTensor* outT, LibTensor* inT,
                         long long divisor, bool signFollowDivisor);

GEN_INSTANCES_INTONLY_OP(template, fwdLibModuloInstThreaded, LibTensor* outT,
                         LibTensor* inT, long long divisor, bool signFollowDivisor,
                         uint64_t flags);
} // namespace dnn_lib
