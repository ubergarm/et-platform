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

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "LibNodes.h"
#include "GenInstances.h"
#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"

using namespace std;

template<typename DstType>
void dnn_lib::fwdLibFusedRowwiseQuantizedSparseLengthsSumInstFloatTyVectorized(
        void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
        void *pdata, void *pdataDims, void *pdataPitches,
        void *pindices, void *plengths, unsigned int pLengthsSize,
        uint64_t flags,
        const uint32_t minionOffset, const uint32_t assignedMinions) {

  const bool float32Dst = (std::is_same<DstType, float>::value);
  const bool float16Dst = (std::is_same<DstType, float16>::value);

  fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized<false, float32Dst, float16Dst>(
    pdst, pdstDims, pdstPitches, pdstDimNum,
    nullptr, nullptr,
    pdata, pdataDims, pdataPitches,
    nullptr, nullptr, nullptr,
    pindices, plengths, pLengthsSize, flags,
    minionOffset, assignedMinions);
}

GEN_INSTANCES_1TYPEFP(template, fwdLibFusedRowwiseQuantizedSparseLengthsSumInstFloatTyVectorized,
            void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
            void *pdata, void *pdataDims, void *pdataPitches,
            void *pindices, void *plengths, unsigned int pLengthsSize,
            uint64_t flags,
            const uint32_t minionOffset = 0, const uint32_t numShires = 0);
