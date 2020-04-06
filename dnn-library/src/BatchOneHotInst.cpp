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

#include "BatchOneHotInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibBatchOneHotInst(void *pdst, void *pdstDims,
                                    void *pdstPitches, void *pdata,
                                    void *pdataDims, void *pdataPitches,
                                    void *pvalues, void *pvaluesDims,
                                    void *pvaluesPitches, void *plengths,
                                    const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibBatchOneHotInst<srcType>(pdst, pdstDims,
                                    pdstPitches, pdata,
                                    pdataDims, pdataPitches,
                                    pvalues, pvaluesDims,
                                    pvaluesPitches, plengths,
                                    scale, offset);
}

template <typename srcType>
void fwdLibBatchOneHotInstThreaded(void *pdst, void *pdstDims,
                                            void *pdstPitches, void *pdata,
                                            void *pdataDims, void *pdataPitches,
                                            void *pvalues, void *pvaluesDims,
                                            void *pvaluesPitches, void *plengths,
                                            const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibBatchOneHotInstThreaded<srcType>(pdst, pdstDims,
                                            pdstPitches, pdata,
                                            pdataDims, pdataPitches,
                                            pvalues, pvaluesDims,
                                            pvaluesPitches, plengths,
                                            scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibBatchOneHotInst, void *pdst, void *pdstDims, void *pdstPitches,
                              void *pdata, void *pdataDims, void *pdataPitches,
                              void *pvalues, void *pvaluesDims, void *pvaluesPitches,
                              void *plengths, const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibBatchOneHotInstThreaded, void *pdst, void *pdstDims, void *pdstPitches,
                                      void *pdata, void *pdataDims, void *pdataPitches,
                                      void *pvalues, void *pvaluesDims, void *pvaluesPitches,
                                      void *plengths, const float *scale, const int32_t *offset, uint64_t flags);
}
