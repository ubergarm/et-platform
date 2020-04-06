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

template <typename srcType>
void fwdLibLocalResponseNormalizationInst(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    unsigned int halfWindowSize, float alpha, float beta, float k, const float *scale,
    const int32_t *offset) {

  dnn_lib::inlining::fwdLibLocalResponseNormalizationInst<srcType>(
    dstMatrix, dstMatrixDims, dstMatrixPitches,
    dst2Matrix, dst2MatrixDims, dst2MatrixPitches,
    activations, activationsDims, activationsPitches,
    halfWindowSize, alpha, beta, k, scale,
    offset);
}

template <typename srcType>
void fwdLibLocalResponseNormalizationInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    unsigned int halfWindowSize, float alpha, float beta, float k, const float *scale,
    const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibLocalResponseNormalizationInstThreaded<srcType>(
    dstMatrix, dstMatrixDims, dstMatrixPitches,
    dst2Matrix, dst2MatrixDims, dst2MatrixPitches,
    activations, activationsDims, activationsPitches,
    halfWindowSize, alpha, beta, k, scale,
    offset, flags);
}

template <typename srcType>
void fwdLibLocalResponseNormalizationInstVectorized(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    unsigned int halfWindowSize, float alpha, float beta, float k, const float *scale,
    const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibLocalResponseNormalizationInstVectorized<srcType>(
    dstMatrix, dstMatrixDims, dstMatrixPitches,
    dst2Matrix, dst2MatrixDims, dst2MatrixPitches,
    activations, activationsDims, activationsPitches,
    halfWindowSize, alpha, beta, k, scale,
    offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibLocalResponseNormalizationInst, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                                             void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches,
                                             void *activations, void *activationsDims, void *activationsPitches,
                                             unsigned int halfWindowSize, float alpha, float beta, float k,
                                             const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibLocalResponseNormalizationInstThreaded, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                                             void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches,
                                             void *activations, void *activationsDims, void *activationsPitches,
                                             unsigned int halfWindowSize, float alpha, float beta, float k,
                                             const float *scale, const int32_t *offset, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibLocalResponseNormalizationInstVectorized, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                                             void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches,
                                             void *activations, void *activationsDims, void *activationsPitches,
                                             unsigned int halfWindowSize, float alpha, float beta, float k,
                                             const float *scale, const int32_t *offset, uint64_t flags);
} // namespace dnn_lib
