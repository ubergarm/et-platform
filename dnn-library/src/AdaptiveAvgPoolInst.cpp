/*-------------------------------------------------------------------------
 * Copyright (C) 2020, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include "AdaptiveAvgPoolInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibAdaptiveAvgPoolInst(void *dstMatrix, void *dstMatrixDims,
                                void *dstMatrixPitches, void *activations,
                                void *activationsDims, void *activationsPitches,
                                const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibAdaptiveAvgPoolInst<srcType>(dstMatrix, dstMatrixDims,
                                dstMatrixPitches, activations,
                                activationsDims, activationsPitches,
                                scale, offset);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibAdaptiveAvgPoolInst,void *dstMatrix, void *dstMatrixDims,
                 void *dstMatrixPitches, void *activations,
                 void *activationsDims, void *activationsPitches,
                 const float *scale, const int32_t *offset);
} // dnn_lib
