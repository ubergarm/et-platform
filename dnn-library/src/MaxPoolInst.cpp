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

#include "MaxPoolInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibMaxPoolInst(bool argMax, void *dstMatrix, void *dstMatrixDims,
                                void *dstMatrixPitches, void *dst2Matrix,
                                void *dst2MatrixDims, void *dst2MatrixPitches, void *srcMatrixPitchesNoPadding,
                                void *activations, void *activationsDims,
                                void *activationsPitches, void *pkernels,
                                void *pstrides, void *ppads, const float *scale,
                                const int32_t *offset) {

  dnn_lib::inlining::fwdLibMaxPoolInst<srcType>(argMax, dstMatrix, dstMatrixDims,
                                dstMatrixPitches, dst2Matrix,
                                dst2MatrixDims, dst2MatrixPitches, srcMatrixPitchesNoPadding,
                                activations, activationsDims,
                                activationsPitches, pkernels,
                                pstrides, ppads, scale,
                                offset);
}

template <typename srcType, typename dstType>
void fwdLibMaxPoolInstThreaded(
    bool argMax, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches, void *srcMatrixPitchesNoPadding,
    void *activations, void *activationsDims, void *activationsPitches,
    void *pkernels, void *pstrides, void *ppads, const float *scale, const int32_t *offset,
    uint64_t flags) {

  dnn_lib::inlining::fwdLibMaxPoolInstThreaded<srcType, dstType>(
    argMax, dstMatrix, dstMatrixDims, dstMatrixPitches,
    dst2Matrix, dst2MatrixDims, dst2MatrixPitches, srcMatrixPitchesNoPadding,
    activations, activationsDims, activationsPitches,
    pkernels, pstrides, ppads, scale, offset,
    flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibMaxPoolInst, bool argMax, void *dstMatrix,void *dstMatrixDims,
                         void *dstMatrixPitches, void *dst2Matrix,
                         void *dst2MatrixDims, void *dst2MatrixPitches, void *srcMatrixPitchesNoPadding,
                         void *activations, void *activationsDims,
                         void *activationsPitches, void *pkernels,
                         void *pstrides, void *ppads, const float *scale,
                         const int32_t *offset);

GEN_INSTANCES_2TYPE_OP(template, fwdLibMaxPoolInstThreaded, bool argMax, void *dstMatrix,void *dstMatrixDims,
                         void *dstMatrixPitches, void *dst2Matrix,
                         void *dst2MatrixDims, void *dst2MatrixPitches, void *srcMatrixPitchesNoPadding,
                         void *activations, void *activationsDims,
                         void *activationsPitches, void *pkernels,
                         void *pstrides, void *ppads, const float *scale,
                         const int32_t *offset, uint64_t flags);
} // namespace dnn_lib
