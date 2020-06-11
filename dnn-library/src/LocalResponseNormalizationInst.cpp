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

#include "LocalResponseNormalizationInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elK>
void fwdLibLocalResponseNormalizationInst(LibTensor* out1T, LibTensor* out2T,
        LibTensor* inT, unsigned int halfWindowSize, float alpha, float beta,
        float k) {

  dnn_lib::inlining::fwdLibLocalResponseNormalizationInst<elK>(out1T, out2T,
                                           inT, halfWindowSize, alpha, beta, k);
}

template <ElemKind elK>
void fwdLibLocalResponseNormalizationInstThreaded(LibTensor* out1T,
    LibTensor* out2T, LibTensor* inT, unsigned int halfWindowSize, float alpha,
    float beta, float k, uint64_t flags) {

  dnn_lib::inlining::fwdLibLocalResponseNormalizationInstThreaded<elK>(
    out1T, out2T, inT, halfWindowSize, alpha, beta, k, flags);
}

template <ElemKind elK>
void fwdLibLocalResponseNormalizationInstVectorized(LibTensor* out1T,
    LibTensor* out2T, LibTensor* inT, unsigned int halfWindowSize, float alpha,
    float beta, float k, uint64_t flags) {

  dnn_lib::inlining::fwdLibLocalResponseNormalizationInstVectorized<elK>(
                out1T, out2T, inT, halfWindowSize, alpha, beta, k, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibLocalResponseNormalizationInst,
                 LibTensor* out1T, LibTensor* out2T, LibTensor* inT,
                 unsigned int halfWindowSize, float alpha, float beta, float k);

GEN_INSTANCES_OP(template, fwdLibLocalResponseNormalizationInstThreaded,
                 LibTensor* out1T, LibTensor* out2T, LibTensor* inT,
                 unsigned int halfWindowSize, float alpha, float beta, float k, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibLocalResponseNormalizationInstVectorized, LibTensor* out1T,
                 LibTensor* out2T, LibTensor* inT, unsigned int halfWindowSize, float alpha,
                 float beta, float k, uint64_t flags);
} // namespace dnn_lib
