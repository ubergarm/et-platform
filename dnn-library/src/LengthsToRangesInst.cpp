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

#include "LengthsToRangesInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibLengthsToRangesInst(void *dstT, void *dstDims,
                                        void *dstPitches, void *plengths,
                                        unsigned int lenDim, const float *scale,
                                        const int32_t *offset) {

  dnn_lib::inlining::fwdLibLengthsToRangesInst<srcType>(dstT, dstDims,
                                        dstPitches, plengths,
                                        lenDim, scale,
                                        offset);
}

template <typename srcType>
void fwdLibLengthsToRangesInstThreaded(void *dstT, void *dstDims,
                                        void *dstPitches, void *plengths,
                                        unsigned int lenDim, const float *scale,
                                        const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibLengthsToRangesInstThreaded<srcType>(dstT, dstDims,
                                        dstPitches, plengths,
                                        lenDim, scale,
                                        offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibLengthsToRangesInst, void *dstT, void *dstDims, void *dstPitches,
                                  void *plengths, unsigned int lenDim,
                                  const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibLengthsToRangesInstThreaded, void *dstT, void *dstDims, void *dstPitches,
                                  void *plengths, unsigned int lenDim,
                                  const float *scale, const int32_t *offset, uint64_t flags);

} // namespace dnn_lib
