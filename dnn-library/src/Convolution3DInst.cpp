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

#include "Convolution3DInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibConvolution3DInst(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    void *pkernels, void *pstrides, void *ppads, unsigned int group,
    const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibConvolution3DInst<srcType>(
    dstMatrix, dstMatrixDims, dstMatrixPitches,
    activations, activationsDims, activationsPitches,
    weights, weightsDims, weightPitches, bias,
    pkernels, pstrides, ppads, group,
    scale, offset);
}

template <typename srcType>
void fwdLibConvolution3DInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    void *pkernels, void *pstrides, void *ppads, unsigned int group,
    const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibConvolution3DInstThreaded<srcType>(
    dstMatrix, dstMatrixDims, dstMatrixPitches,
    activations, activationsDims, activationsPitches,
    weights, weightsDims, weightPitches, bias,
    pkernels, pstrides, ppads, group,
    scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibConvolution3DInst, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                                void *activations, void *activationsDims, void *activationsPitches,
                                void *weights, void *weightsDims, void *weightPitches, void *bias,
                                void *pkernels, void *pstrides, void *ppads, unsigned int group,
                                const float *scale, const int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibConvolution3DInstThreaded, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                                void *activations, void *activationsDims, void *activationsPitches,
                                void *weights, void *weightsDims, void *weightPitches, void *bias,
                                void *pkernels, void *pstrides, void *ppads, unsigned int group,
                                const float *scale, const int32_t *offset, uint64_t flags);

} // namespace dnn_lib
