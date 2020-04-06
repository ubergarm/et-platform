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

#include "FullyConnectedInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibFullyConnectedInst(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibFullyConnectedInst<srcType>(
    dstMatrix, dstMatrixDims, dstMatrixPitches,
    activations, activationsDims, activationsPitches,
    weights, weightsDims, weightPitches, bias,
    scale, offset);
}

template <typename srcType>
void fwdLibFullyConnectedInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibFullyConnectedInstThreaded<srcType>(
    dstMatrix, dstMatrixDims, dstMatrixPitches,
    activations, activationsDims, activationsPitches,
    weights, weightsDims, weightPitches, bias,
    scale, offset, flags);
}

template <typename src1Type, typename src2Type, typename dstType>
void fwdLibFullyConnectedInstVectorized(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches1,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibFullyConnectedInstVectorized<src1Type, src2Type, dstType>(
    dstMatrix, dstMatrixDims, dstMatrixPitches1,
    activations, activationsDims, activationsPitches,
    weights, weightsDims, weightPitches, bias,
    scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibFullyConnectedInst, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                                      void *activations, void *activationsDims, void *activationsPitches,
                                      void *weights, void *weightsDims, void *weightPitches,
                                      void *bias, const float *scale, const int32_t *offset );

GEN_INSTANCES_OP(template, fwdLibFullyConnectedInstThreaded, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                                      void *activations, void *activationsDims, void *activationsPitches,
                                      void *weights, void *weightsDims, void *weightPitches,
                                      void *bias, const float *scale, const int32_t *offset, uint64_t flags );

GEN_INSTANCES_3TYPE_OP(template, fwdLibFullyConnectedInstVectorized, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                                      void *activations, void *activationsDims, void *activationsPitches,
                                      void *weights, void *weightsDims, void *weightPitches,
                                      void *bias, const float *scale, const int32_t *offset, uint64_t flags );

} // namespace dnn_lib
