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

template <typename srcType, typename opType>
void fwdLibElementSingleInst(LibTensor* outT, LibTensor* inT) {

  dnn_lib::inlining::fwdLibElementSingleInst<srcType, opType>(outT, inT);
}

template <typename srcType, typename opType>
void fwdLibElementSingleInstThreaded(LibTensor* outT, LibTensor* inT, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementSingleInstThreaded<srcType, opType>(outT, inT, flags);
}

template <typename src1Type, typename dstType, typename opType>
void fwdLibElementSingleInstVectorized(LibTensor* outT, LibTensor* inT,
                                       const float* scaleA,
                                       const int32_t* offsetA, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementSingleInstVectorized<src1Type, dstType, opType>(outT, inT,
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
