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

#include "FusedRowwiseQuantizedSparseLengthsWeightedSum.h" // From include/inlining

namespace dnn_lib {

void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize) {

  dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    pdst, pdstDims, pdstPitches, pdstDimNum,
    pdata, pdataDims, pdataPitches, pweights,
    pweightsDims, pweightsPitches, pindices, plengths,
    pLengthsSize);
}

void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
        void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
        void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
        void *pweightsDims, void *pweightsPitches, void *pindices,
        void *plengths, unsigned int pLengthsSize, uint64_t flags) {

  dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
        pdst, pdstDims, pdstPitches, pdstDimNum,
        pdata, pdataDims, pdataPitches, pweights,
        pweightsDims, pweightsPitches, pindices,
        plengths, pLengthsSize, flags);
}

template <bool Weighted, bool Float32Dst, bool Float16Dst>
inline void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumVect(
    uintptr_t minionCurrIndex, uintptr_t currSegmentLength,
    uint8_t *tAInput, int64_t *indices, uintptr_t dataRowPitch,
    uintptr_t dataRowSize, uintptr_t dstElemSize,
    uint8_t *tWInput, uint8_t *dst_ptr, uint8_t *dst2_ptr) { 

  dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumVect<Weighted, Float32Dst, Float16Dst>(
    minionCurrIndex, currSegmentLength,
    tAInput, indices, dataRowPitch,
    dataRowSize, dstElemSize,
    tWInput, dst_ptr, dst2_ptr);
}

template<bool Weighted, bool Float32Dst, bool Float16Dst>
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized(
        LibTensor* outT, LibTensor* out2T, LibTensor* in1T, LibTensor* in2T,
        LibTensor* in3T, LibTensor* in4T, uint64_t flags,
        const uint32_t minionOffset, const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized<Weighted, Float32Dst, Float16Dst> (
        outT, out2T, in1T, in2T, in3T, in4T,
        flags, minionOffset, assignedMinions);
}

template<typename Type>
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized(
        LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
        uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized<Type>(
        outT, in1T, in2T, in3T, in4T, flags, minionOffset, assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_1TYPEFP(template, fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized,
        LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
        uint64_t flags, const uint32_t minionOffset, const uint32_t numShires);

GEN_INSTANCES_FRQSLWS_V(template, fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized,
                        LibTensor* outT, LibTensor* out2T, LibTensor* in1T, LibTensor* in2T,
                        LibTensor* in3T, LibTensor* in4T, uint64_t flags,
                        const uint32_t minionOffset, const uint32_t numShires);

GEN_INSTANCES_FRQSLWS_V(template, fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumVect,
                        uintptr_t minionCurrIndex, uintptr_t currSegmentLength,
                        uint8_t *tAInput, int64_t *indices, uintptr_t dataRowPitch,
                        uintptr_t dataRowSize, uintptr_t dstElemSize,
                        uint8_t *tWInput, uint8_t *dst_ptr, uint8_t *dst2_ptr);
} // namespace dnn_lib
