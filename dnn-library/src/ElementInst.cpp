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

#include "ElementInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType, typename opType>
void fwdLibElementInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T) {

  dnn_lib::inlining::fwdLibElementInst<srcType, opType>(outT, in1T, in2T);

}

template <typename src1Type, typename src2Type, typename dstType, typename opType>
void fwdLibElementInstThreaded(LibTensor* outT, LibTensor* in1T,
                               LibTensor* in2T, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementInstThreaded<src1Type, src2Type, dstType, opType>(
                                                             outT, in1T, in2T, flags);
}

template <typename src1Type, typename src2Type, typename dstType, typename opType>
void fwdLibElementInstVectorized(LibTensor* outT, LibTensor* in1T,
                                 LibTensor* in2T, const float* scale,
                                 const int32_t* offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibElementInstVectorized<src1Type, src2Type, dstType, opType>(
                                             outT, in1T, in2T, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Add, LibTensor* outT,
                        LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Sub, LibTensor* outT,
                        LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Div, LibTensor* outT,
                        LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Mul, LibTensor* outT,
                        LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Max, LibTensor* outT,
                        LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Min, LibTensor* outT,
                        LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_INSTANCES(template, fwdLibElementInst,Pow, LibTensor* outT,
                        LibTensor* in1T, LibTensor* in2T);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Add, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Sub, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Div, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Mul, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Max, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Min, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstThreaded,Pow, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Add, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, const float* scale, const int32_t* offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Sub, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, const float* scale, const int32_t* offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Div, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, const float* scale, const int32_t* offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Mul, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, const float* scale, const int32_t* offset, uint64_t flags);
  
GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Max, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, const float* scale, const int32_t* offset, uint64_t flags);
  
GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Min, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, const float* scale, const int32_t* offset, uint64_t flags);

GEN_INSTANCES_3TYPE(template, fwdLibElementInstVectorized,Pow, LibTensor* outT,
                    LibTensor* in1T, LibTensor* in2T, const float* scale, const int32_t* offset, uint64_t flags);
}
