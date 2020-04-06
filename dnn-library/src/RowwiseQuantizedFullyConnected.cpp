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

#include "RowwiseQuantizedFullyConnected.h" // From include/inlining

namespace dnn_lib {

void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTy(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTy(
    pdst, pdstDims, pdstPitches, pdata, pdataDims,
    pdataPitches, pscale, poffset, pweights,
    pweightsDims, pweightsPitches, pbias, srcscale,
    srcoffset, dstscale, dstoffset, biasscale,
    biasoffset);
}

void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyThreaded(
    pdst, pdstDims, pdstPitches, pdata, pdataDims,
    pdataPitches, pscale, poffset, pweights,
    pweightsDims, pweightsPitches, pbias, srcscale,
    srcoffset, dstscale, dstoffset, biasscale,
    biasoffset, flags);
}

void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyVectorized(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyVectorized(
    pdst, pdstDims, pdstPitches, pdata, pdataDims,
    pdataPitches, pscale, poffset, pweights,
    pweightsDims, pweightsPitches, pbias, srcscale,
    srcoffset, dstscale, dstoffset, biasscale,
    biasoffset, flags);
}

void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyAligned32Bytes(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyAligned32Bytes(
    pdst, pdstDims, pdstPitches, pdata, pdataDims,
    pdataPitches, pscale, poffset, pweights,
    pweightsDims, pweightsPitches, pbias, srcscale,
    srcoffset, dstscale, dstoffset, biasscale,
    biasoffset, flags);
}

} // namespace dnn_lib
