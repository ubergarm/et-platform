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
void fwdLibMatMulInst(void *dstMatrix, void *dstMatrixDims,
                               void *dstMatrixPitches, void *activations,
                               void *activationsDims, void *activationsPitches,
                               void *weights, void *weightsDims,
                               void *weightPitches, const float *scale,
                               const int32_t *offset) {

  dnn_lib::inlining::fwdLibMatMulInst<srcType>(dstMatrix, dstMatrixDims,
                               dstMatrixPitches, activations,
                               activationsDims, activationsPitches,
                               weights, weightsDims,
                               weightPitches, scale,
                               offset);
}

template <typename srcType>
void fwdLibMatMulInstThreaded(void *dstMatrix, void *dstMatrixDims,
                                       void *dstMatrixPitches,
                                       void *activations, void *activationsDims,
                                       void *activationsPitches, void *weights,
                                       void *weightsDims, void *weightPitches,
                                       const float *scale, const int32_t *offset,
                                       uint64_t flags,
                                       const uint32_t minionOffset,
                                       const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibMatMulInstThreaded<srcType>(dstMatrix, dstMatrixDims,
                                       dstMatrixPitches,
                                       activations, activationsDims,
                                       activationsPitches, weights,
                                       weightsDims, weightPitches,
                                       scale, offset,
                                       flags,
                                       minionOffset,
                                       assignedMinions);
}


template <typename srcType>
void fwdLibMatMulInstVectorized(void *dstMatrix, void *dstMatrixDims,
                                         void *dstMatrixPitches,
                                         void *activations, void *activationsDims,
                                         void *activationsPitches, void *weights,
                                         void *weightsDims, void *weightPitches,
                                         const float *scale, const int32_t *offset, uint64_t flags,
                                         const uint32_t minionOffset,
                                         const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibMatMulInstVectorized<srcType>(dstMatrix, dstMatrixDims,
                                         dstMatrixPitches,
                                         activations, activationsDims,
                                         activationsPitches, weights,
                                         weightsDims, weightPitches,
                                         scale, offset, flags,
                                         minionOffset,
                                         assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibMatMulInst, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibMatMulInstThreaded, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         const float *scale, const int32_t *offset, uint64_t flags,
                         const uint32_t minionOffset = 0, const uint32_t numShires = 0);

GEN_INSTANCES_OP(template, fwdLibMatMulInstVectorized, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         const float *scale, const int32_t *offset, uint64_t flags,
                         const uint32_t minionOffset = 0, const uint32_t numShires = 0);

} // namespace dnn_lib
