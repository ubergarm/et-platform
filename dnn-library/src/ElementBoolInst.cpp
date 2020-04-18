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

#include "ElementBoolInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType, typename opType>
void fwdLibElementBoolInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T) {

  dnn_lib::inlining::fwdLibElementBoolInst<srcType, opType>(outT, in1T, in2T);
}

template <typename srcType, typename opType>
void fwdLibElementBoolInstThreaded(LibTensor* outT, LibTensor* in1T,
                                   LibTensor* in2T, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementBoolInstThreaded<srcType, opType>(outT, in1T,
                                                                    in2T, flags);
}

template <typename src1Type, typename src2Type, typename opType>
void fwdLibElementBoolInstVectorized(LibTensor* outT, LibTensor* in1T,
                                     LibTensor* in2T, const float* scale,
                                     const int32_t* offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementBoolInstVectorized<src1Type, src2Type, opType>(
                                                     outT, in1T, in2T, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInst,CmpEQ, LibTensor* outT,
                        LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInst,CmpLTE, LibTensor* outT,
                        LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInst,CmpLT, LibTensor* outT,
                        LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInstThreaded,CmpEQ,
                        LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                        uint64_t flags);

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInstThreaded,CmpLTE,
                        LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                        uint64_t flags);

GEN_INSTANCES_INSTANCES(template, fwdLibElementBoolInstThreaded,CmpLT,
                        LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                        uint64_t flags);


GEN_INSTANCES_2TYPE(template, fwdLibElementBoolInstVectorized,CmpEQ,
                    LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                    const float* scale, const int32_t* offset, uint64_t flags);

GEN_INSTANCES_2TYPE(template, fwdLibElementBoolInstVectorized,CmpLTE,
                    LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                    const float* scale, const int32_t* offset, uint64_t flags);
  
GEN_INSTANCES_2TYPE(template, fwdLibElementBoolInstVectorized,CmpLT,
                    LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                    const float* scale, const int32_t* offset, uint64_t flags);  

} // namespace dnn_lib
