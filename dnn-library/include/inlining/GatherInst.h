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

#ifndef _GATHER_INST_H_
#define _GATHER_INST_H_

#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>
#include "Float16.h"
#include "Addresser.h" // From include/internal path
#include "utils.h" // From include/internal path
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {


// The threaded version of the GatherInst function generalises the function to
// any given dimensions for the two source tensors (tInput and tIndices). tInput
// dims: d0 x ··· x dn. batchedDims: i (number between 0 and n). tIndices dims:
// i0 x ··· x ik. Then, the tOutput tensor dimensions are determined. tOutput
// dims: d0 x ··· x d(i-1) x i0 x ··· x ik x d(i+1) x ··· x dn (dstDimsNum =
// srcDimsNum + indicesDimsNum - 1). The GatherInst function consists in copying
// the source tensor's elements in the following way:
// tOutput(x0,...,x(i-1),y0,...,yk,x(i+1),...,xn) =
// tInput(x0,...,x(i-1),tIndices(y0,...,yk),x(i+1),...,xn). The elements in the
// tIndices tensor must be integers between 0 and di - 1, so they are valid
// index values for the i-th dimension of the source tensor tInput.

template <ElemKind srcElK, ElemKind indexElK>
INLINE_ATTR void fwdLibGatherInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, unsigned int batchedDims,
                                  uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<srcElK>::type;
  //  using indexType = typename elemKind2elemTy<indexElK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT->dst  in1T--> src  in2T--> index*/
  void* dstT = outT->getRawDataPointer();
  void* srcT = in1T->getRawDataPointer();
  void* indexT = in2T->getRawDataPointer();

  Addresser<srcElK> tOutput(dstT, outT->getScale(), outT->getOffset());
  const Addresser<srcElK> tInput(srcT, in1T->getScale(), in1T->getOffset());
  const Addresser<indexElK> tIndices(indexT, in2T->getScale(), in2T->getOffset());
  
  // unsigned int *srcIndex = (unsigned int *)srcDims;
  const dim_t *srcIndex = in1T->dims().data();
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *indicesIndex = (unsigned int *)indicesDims;
  const dim_t *indicesIndex = in2T->dims().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = in1T->strides().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *indicesPitch = (unsigned int *)pindicesPitches;
  const dim_t *indicesPitch = in2T->strides().data();

  dim_t srcDimsNum = in1T->ndims();
  dim_t indicesDimsNum = in2T->ndims();

  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  size_t dstDimsNum = srcDimsNum + indicesDimsNum - 1;
  dim_array_t coordOut = {0};
  dim_t last_non_zero_coord;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimsNum, dstPitch,
                           dstIndex, last_non_zero_coord);

  size_t offsetOut = 0;
  for (dim_t i = 0; i < last_non_zero_coord; i++)
    offsetOut += dstPitch[i] * coordOut[i];
  size_t offsetIndices = 0;
  for (dim_t i = 0; i < indicesDimsNum; i++)
    offsetIndices += coordOut[batchedDims + i] * indicesPitch[i];

  size_t offsetIn = 0;
  for (dim_t i = 0; i < batchedDims; i++)
    offsetIn += srcPitch[i] * coordOut[i];
  auto index = tIndices[offsetIndices];
  offsetIn += srcPitch[batchedDims] * index;
  for (dim_t i = batchedDims + 1; i < srcDimsNum; i++)
    offsetIn +=
        srcPitch[i] *
        coordOut[indicesDimsNum + i - 1]; // could iterate just until the last
                                          // source non-zero coordInate. todo

  auto posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    tOutput[offsetOut] = tInput[offsetIn];
    // Coordinates are updated to the next position that must be copied.
    for (size_t j = dstDimsNum - 1; j < dstDimsNum;
         j--) { // the loop goes from j=Dims-1 to 0, and the wraps around => j > dstDimsNum and ends
      if (coordOut[j] != (dstIndex[j] - 1)) {
        offsetOut += dstPitch[j];
        coordOut[j]++;
        if (j >= batchedDims + indicesDimsNum)
          offsetIn += srcPitch[j - indicesDimsNum + 1];
        else if (batchedDims <= j) {
          offsetIndices += indicesPitch[j - batchedDims];
          offsetIn += (tIndices[offsetIndices] - index) * srcPitch[batchedDims];
          index = tIndices[offsetIndices];
        } else
          offsetIn += srcPitch[j];
        break; // Once the coordinates have been updated, a new copy can be
               // performed.
      } else if (j != 0) {
        offsetOut -= (dstIndex[j] - 1) * dstPitch[j];
        coordOut[j] = 0;
        if (j >= batchedDims + indicesDimsNum) {
          auto k = j - indicesDimsNum + 1;
          offsetIn -= (srcIndex[k] - 1) * srcPitch[k];
        } else if (batchedDims <= j) {
          offsetIndices -= (indicesIndex[j - batchedDims] - 1) *
                           indicesPitch[j - batchedDims];
          offsetIn += (tIndices[offsetIndices] - index) * srcPitch[batchedDims];
          index = tIndices[offsetIndices];
        } else
          offsetIn += srcPitch[j];
      } else
        done = true; // The end of the destination tensor has been reached.
    }
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _GATHER_INST_H_
