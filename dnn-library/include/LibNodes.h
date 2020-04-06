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

#ifndef LIBNODES_H_
#define LIBNODES_H_

#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

#include "Float16.h"

#include "Operators.h"

#include "inlining.h"

namespace dnn_lib {

enum class PrecisionMode {
  // TODO: Get same enumerate as Jitter
  PM_FP_32 = 0,   // fp32
  PM_FP_16 = 1,   // fp16
  PM_INT_32 = 2,  // quant int32
  PM_INT_8 = 3,   // quant int8
  PM_INT_16 = 4,  // quant int16
  PM_INT_I32 = 5, // idx int32
  PM_INT_I64 = 6, // idx int64
  PM_UINT_8 = 7,  // quant uint8
  PM_BOOL = 8,    // bool
  MAX_PRECISION_MODES
};

#include "AutoGenInstan.def"

void fwdLibBatchedAddInsti8i32(void *pdst, void *pdstDims, void *pdstPitches,
                               void *pbatch, void *pbatchDims,
                               void *pbatchPitches, unsigned int pbatchDimNum,
                               void *pslice, void *pslicePitches, const float *scale,
                               const int32_t * offset);
void fwdLibBatchedReduceAddInstInt8(void *pdst, void *pdstDims,
                                    void *pdstPitches, void *pbatch,
                                    void *pbatchDims, void *pbatchPitches,
                                    unsigned int pbatchDimNum,
                                    unsigned int axis, const float *scale,
                                    const int32_t * offset);
void fwdLibBatchedReduceAddInstInt8Threaded(void *pdst, void *pdstDims,
                                            void *pdstPitches, void *pbatch,
                                            void *pbatchDims, void *pbatchPitches,
                                            unsigned int pbatchDimNum,
                                            unsigned int axis, const float *scale,
                                            const int32_t * offset, uint64_t flags);
void fwdLibSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize);
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize);
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize);
void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTy(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset);
void fwdLibIntLookupTableInstInt8QTy(void *dstT, void *dstDims,
                                     void *dstPitches, unsigned int dstDimNum,
                                     void *src1T, void *src1Dims,
                                     void *src1Pitches, void *src2T,
                                     void *src2Dims, void *src2Pitches);
void fwdLibFlushL3(uint32_t numShires);
void fwdLibEmbeddingBagInstFloatTy(void *pdst,
                                   void *pdata, uint64_t dataDim1Pitch,
                                   void *pweights,
                                   void *pindices, uint64_t indicesSize,
                                   void *poffsets, uint64_t offsetsSize);
                                
/**********************
 * THREADED FUNCTIONS *
 **********************/
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, uint64_t flags);
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags);
void fwdLibIntLookupTableInstInt8QTyThreaded(
    void *dstT, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src1T, void *src1Dims, void *src1Pitches, void *src2T, void *src2Dims,
    void *src2Pitches, uint64_t flags);
void fwdLibBatchedAddInsti8i32Threaded(void *pdst, void *pdstDims, void *pdstPitches,
                               void *pbatch, void *pbatchDims,
                               void *pbatchPitches, unsigned int pbatchDimNum,
                               void *pslice, void *pslicePitches, const float *scale,
                               const int32_t * offset, uint64_t flags);
void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags);
/************************
 * VECTORIZED FUNCTIONS *
 ************************/
void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyVectorized(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags);
void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyAligned32Bytes(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags);

} // namespace dnn_lib

#endif /* LIBNODES_H_ */
