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

#include "SparseLengthsWeightedSumInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibSparseLengthsWeightedSumInst(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInst<srcType>(
    pdst, pdstDims, pdstPitches, pdstDimNum,
    pdata, pdataDims, pdataPitches, pweights,
    pweightsDims, pweightsPitches, pindices, plengths,
    pLengthsSize, scale, offset);
}

template <typename srcType>
void fwdLibSparseLengthsWeightedSumInstThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInstThreaded<srcType>(
    pdst, pdstDims, pdstPitches, pdstDimNum,
    pdata, pdataDims, pdataPitches, pweights,
    pweightsDims, pweightsPitches, pindices, plengths,
    pLengthsSize, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSparseLengthsWeightedSumInst,void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
                                          void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
                                          void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
                                          unsigned int pLengthsSize, const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibSparseLengthsWeightedSumInstThreaded,void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
                                          void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
                                          void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
                                          unsigned int pLengthsSize, const float *scale, const int32_t *offset, uint64_t flags);

} // namespace dnn_lib
