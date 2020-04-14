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

#include "MatMulInst.h" // From include/inlining path

namespace dnn_lib {

template <typename srcType>
void fwdLibMatMulInst(LibTensor* inT, LibTensor* weighT, LibTensor* outT) {

  dnn_lib::inlining::fwdLibMatMulInst<srcType>(inT, weighT, outT);
}

template <typename srcType>
void fwdLibMatMulInstThreaded(LibTensor* inT, LibTensor* weighT, LibTensor* outT,
                              uint64_t flags, const uint32_t minionOffset,
                              const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibMatMulInstThreaded<srcType>(inT, weighT, outT, flags,
                                                       minionOffset, assignedMinions);
}


template <typename srcType>
void fwdLibMatMulInstVectorized(LibTensor* inT, LibTensor* weighT, LibTensor* outT,
                                uint64_t flags, const uint32_t minionOffset,
                                const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibMatMulInstVectorized<srcType>(inT, weighT, outT,flags,
                                                         minionOffset, assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibMatMulInst, LibTensor* inT, LibTensor* weighT,
                 LibTensor* outT);

GEN_INSTANCES_OP(template, fwdLibMatMulInstThreaded, LibTensor* inT,
                 LibTensor* weighT, LibTensor* outT,  uint64_t flags,
                 const uint32_t minionOffset = 0, const uint32_t numShires = 0);

GEN_INSTANCES_OP(template, fwdLibMatMulInstVectorized,  LibTensor* inT,
                 LibTensor* weighT, LibTensor* outT, uint64_t flags,
                 const uint32_t minionOffset = 0, const uint32_t numShires = 0);

} // namespace dnn_lib
