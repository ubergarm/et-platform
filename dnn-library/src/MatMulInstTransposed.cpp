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

#include "MatMulInstTransposed.h"

namespace dnn_lib {

template <typename srcType>
void fwdLibMatMulInstTransposed(void *dstMatrix, void *dstMatrixDims,
                                         void *dstMatrixPitches, void *activations,
                                         void *activationsDims, void *activationsPitches,
                                         void *weights, void *weightsDims,
                                         void *weightPitches, const float *scale,
                                         const int32_t *offset) {

  dnn_lib::inlining::fwdLibMatMulInstTransposed<srcType>(dstMatrix, dstMatrixDims,
                                         dstMatrixPitches, activations,
                                         activationsDims, activationsPitches,
                                         weights, weightsDims,
                                         weightPitches, scale,
                                         offset);
}

template <typename srcType>
void fwdLibMatMulInstThreadedTransposed(void *dstMatrix, void *dstMatrixDims,
                                                 void *dstMatrixPitches,
                                                 void *activations, void *activationsDims,
                                                 void *activationsPitches, void *weights,
                                                 void *weightsDims, void *weightPitches,
                                                 const float *scale, const int32_t *offset,
                                                 uint64_t flags) {

  dnn_lib::inlining::fwdLibMatMulInstThreadedTransposed<srcType>(dstMatrix, dstMatrixDims,
                                                 dstMatrixPitches,
                                                 activations, activationsDims,
                                                 activationsPitches, weights,
                                                 weightsDims, weightPitches,
                                                 scale, offset,
                                                 flags);
}

template <typename srcType>
void fwdLibMatMulInstVectorizedTransposed(void *dstMatrix, void *dstMatrixDims,
                                                   void *dstMatrixPitches,
                                                   void *activations, void *activationsDims,
                                                   void *activationsPitches, void *weights,
                                                   void *weightsDims, void *weightPitches,
                                                   const float *scale, const int32_t *offset,
                                                   uint64_t flags) {

  dnn_lib::inlining::fwdLibMatMulInstVectorizedTransposed<srcType>(dstMatrix, dstMatrixDims,
                                                   dstMatrixPitches,
                                                   activations, activationsDims,
                                                   activationsPitches, weights,
                                                   weightsDims, weightPitches,
                                                   scale, offset,
                                                   flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibMatMulInstTransposed, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibMatMulInstThreadedTransposed, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         const float *scale, const int32_t *offset, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibMatMulInstVectorizedTransposed, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                         void *activations, void *activationsDims, void *activationsPitches,
                         void *weights, void *weightsDims, void *weightPitches,
                         const float *scale, const int32_t *offset, uint64_t flags);

} // namespace dnn_lib
