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

#include "SparseToDenseMaskInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibSparseToDenseMaskInst(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pdefault,
    int pdefaultSize, void *pindices, void *plengths, unsigned int pLengthsSize,
    void *pmask, unsigned int pMaskSize, const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibSparseToDenseMaskInst<srcType>(
    pdst, pdstDims, pdstPitches, pdstDimNum,
    pdata, pdataDims, pdataPitches, pdefault,
    pdefaultSize, pindices, plengths, pLengthsSize,
    pmask, pMaskSize, scale, offset);
}

template <typename srcType>
void fwdLibSparseToDenseMaskInstThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, unsigned int pdataDimNum, void *pdefault,
    int pdefaultSize, void *pindices, void *plengths, unsigned int pLengthsSize,
    void *pmask, unsigned int pMaskSize, const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibSparseToDenseMaskInstThreaded<srcType>(
    pdst, pdstDims, pdstPitches, pdstDimNum,
    pdata, pdataDims, pdataPitches, pdataDimNum, pdefault,
    pdefaultSize, pindices, plengths, pLengthsSize,
    pmask, pMaskSize, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSparseToDenseMaskInst, void *pdst, void *pdstDims, void *pdstPitches,
                                    unsigned int pdstDimNum, void *pdata, void *pdataDims,
                                    void *pdataPitches, void *pdefault, int pdefaultSize,
                                    void *pindices, void *plengths, unsigned int pLengthsSize,
                                    void *pmask, unsigned int pMaskSize, const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibSparseToDenseMaskInstThreaded, void *pdst, void *pdstDims, void *pdstPitches,
                                    unsigned int pdstDimNum, void *pdata, void *pdataDims,
                                    void *pdataPitches, unsigned int pdataDimNum, void *pdefault, int pdefaultSize,
                                    void *pindices, void *plengths, unsigned int pLengthsSize,
                                    void *pmask, unsigned int pMaskSize, const float *scale, const int32_t *offset, uint64_t flags);

} // namespace dnn_lib
