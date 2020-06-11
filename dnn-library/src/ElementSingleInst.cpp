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

#include "ElementSingleInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind dstElk, ElemKind srcElk, typename opType>
void fwdLibElementSingleInst(LibTensor* outT, LibTensor* inT) {

  dnn_lib::inlining::fwdLibElementSingleInst<dstElk, srcElk, opType>(outT, inT);
}

template <ElemKind dstElk, ElemKind srcElk, typename opType>
void fwdLibElementSingleInstThreaded(LibTensor* outT, LibTensor* inT, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementSingleInstThreaded<dstElk, srcElk, opType>(outT, inT, flags);
}

template <ElemKind dstElk, ElemKind srcElk, typename opType>
void fwdLibElementSingleInstVectorized(LibTensor* outT, LibTensor* inT,
                                       const float* scaleA,
                                       const int32_t* offsetA, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementSingleInstVectorized<dstElk, srcElk, opType>(outT, inT,
                                                                               scaleA, offsetA, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_INSTANCES(template, fwdLibElementSingleInst,ElementLog,
                        LibTensor* outT, LibTensor* inT);

GEN_INSTANCES_INSTANCES(template, fwdLibElementSingleInstThreaded,ElementLog,
                        LibTensor* outT, LibTensor* inT, uint64_t flags);

GEN_INSTANCES_2TYPE(template, fwdLibElementSingleInstVectorized,ElementLog,
                    LibTensor* outT, LibTensor* inT, const float* scaleA,
                    const int32_t* offsetA, uint64_t flags);

} // namespace dnn_lib
