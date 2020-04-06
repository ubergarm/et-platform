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

#include "RowwiseQuantizedSparseLengthsWeightedSum.h" // From include/inlining

namespace dnn_lib {

void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    pdst, pdstDims, pdstPitches, pdstDimNum,
    pdata, pdataDims, pdataPitches, pscale,
    poffset, pweights, pweightsDims, pweightsPitches,
    pindices, plengths, pLengthsSize);
}

void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
    pdst, pdstDims, pdstPitches, pdstDimNum,
    pdata, pdataDims, pdataPitches, pscale,
    poffset, pweights, pweightsDims, pweightsPitches,
    pindices, plengths, pLengthsSize, flags);
}

template<bool Int8Src, bool Float16Dst>
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized(
        void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
	    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
	    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
	    void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags,
	    const uint32_t minionOffset, const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized<Int8Src, Float16Dst>(
        pdst, pdstDims, pdstPitches, pdstDimNum,
	    pdata, pdataDims, pdataPitches, pscale,
	    poffset, pweights, pweightsDims, pweightsPitches,
	    pindices, plengths, pLengthsSize, flags,
	    minionOffset, assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_RQSLWS_V(template, fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized,
            	void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
	    		void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
	    		void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
	    		void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags,
	    		const uint32_t minionOffset, const uint32_t assignedMinions);

} // namespace dnn_lib
