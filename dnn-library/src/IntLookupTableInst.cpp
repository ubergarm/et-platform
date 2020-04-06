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

#include "IntLookupTableInst.h" // From include/inlining

namespace dnn_lib {

void fwdLibIntLookupTableInstInt8QTy(
    void *dstT, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src1T, void *src1Dims, void *src1Pitches, void *src2T, void *src2Dims,
    void *src2Pitches) {

  dnn_lib::inlining::fwdLibIntLookupTableInstInt8QTy(
    dstT, dstDims, dstPitches, dstDimNum,
    src1T, src1Dims, src1Pitches, src2T, src2Dims,
    src2Pitches);
}

void fwdLibIntLookupTableInstInt8QTyThreaded(
    void *dstT, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src1T, void *src1Dims, void *src1Pitches, void *src2T, void *src2Dims,
    void *src2Pitches, uint64_t flags) {

  dnn_lib::inlining::fwdLibIntLookupTableInstInt8QTyThreaded(
    dstT, dstDims, dstPitches, dstDimNum,
    src1T, src1Dims, src1Pitches, src2T, src2Dims,
    src2Pitches, flags);
}

} // namespace dnn_lib
