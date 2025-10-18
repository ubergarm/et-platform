/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _GATHER_INST_H_
#define _GATHER_INST_H_

#include "Float16.h"
#include "LibTensor.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

// The threaded version of the GatherInst function generalises the function to
// any given dimensions for the two source tensors (Src and Indices). Src
// dims: d0 x ··· x dn. batchedDims: i (number between 0 and n). Indices dims:
// i0 x ··· x ik. Then, the output tensor dimensions are determined. Dst
// dims: d0 x ··· x d(i-1) x i0 x ··· x ik x d(i+1) x ··· x dn (dstDimsNum =
// srcDimsNum + indicesDimsNum - 1). The GatherInst function consists in copying
// the source tensor's elements in the following way:
// Dst(x0,...,x(i-1),y0,...,yk,x(i+1),...,xn) =
// Src(x0,...,x(i-1),Indices(y0,...,yk),x(i+1),...,xn). The elements in the
// Indices tensor must be integers between -di and di - 1, so they are valid
// index values for the i-th dimension of the source tensor Src.

template <ElemKind srcElK, ElemKind indexElK>
INLINE_ATTR void fwdLibGatherInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, unsigned int batchedDims,
                                  uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<srcElK>::type;
  using idxType = typename elemKind2elemTy<indexElK>::type;

  assert(get_minion_id() >= minionOffset);
  assert((minionOffset == 0) or (minionOffset + assignedMinions < MIN_PER_SHIRE * activeShires(flags)));
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  auto dstT = outT->getRawDataPointer<uint8_t>();
  auto srcT = in1T->getRawDataPointer<uint8_t>();
  auto indexT = in2T->getRawDataPointer<idxType>();

  const dim_t *srcIndex = in1T->dims().data();
  const dim_t *dstIndex = outT->dims().data();
  const dim_t *indicesIndex = in2T->dims().data();
  const dim_t *srcPitch = in1T->strides().data();
  const dim_t *dstPitch = outT->strides().data();
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
  auto index = indexT[offsetIndices];
  index = (index < 0) ? index + srcIndex[batchedDims] : index;
  offsetIn += srcPitch[batchedDims] * index;
  for (dim_t i = batchedDims + 1; i < srcDimsNum; i++)
    offsetIn +=
        srcPitch[i] *
        coordOut[indicesDimsNum + i - 1]; // could iterate just until the last
                                          // source non-zero coordInate. todo
  size_t posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    // Copy chunks of consecutive elements (last dim) until the end of the dimension
    size_t copySize =
      (batchedDims + indicesDimsNum < dstDimsNum - 1) ? dstIndex[dstDimsNum - 1] - coordOut[dstDimsNum - 1] : 1;
    if (offsetOut + copySize > posMax) {
      // Make sure posMax is not surpassed
      copySize = posMax - offsetOut;
    }
    copyBytes(&srcT[offsetIn * typeSize], &dstT[offsetOut * typeSize], copySize * typeSize);
    // Coordinates are updated to the next position that must be copied.
    for (int j = dstDimsNum - 1; j >= 0; j--) {
      if (coordOut[j] + copySize != dstIndex[j]) {
        offsetOut += dstPitch[j] * copySize;
        coordOut[j] += copySize;
        if (j >= (int)(batchedDims + indicesDimsNum)) {
          offsetIn += srcPitch[j - indicesDimsNum + 1] * copySize;
        } else if ((int)batchedDims <= j) {
          offsetIndices += indicesPitch[j - batchedDims];
          auto index_next = indexT[offsetIndices];
          index_next = (index_next < 0) ? index_next + srcIndex[batchedDims] : index_next;
          offsetIn += (index_next - index) * srcPitch[batchedDims];
          index = index_next;
        } else {
          offsetIn += srcPitch[j];
        }
        break; // Once the coordinates have been updated, a new copy can be performed.
      } else if (j != 0) {
        // previous iteration (of the while loop) was in last element of dest dimension j, reset that dimension for the
        // destination, the source data and the indices.
        offsetOut -= (dstIndex[j] - copySize) * dstPitch[j];
        coordOut[j] = 0;
        if (j >= (int)(batchedDims + indicesDimsNum)) {
          auto k = j - indicesDimsNum + 1;
          offsetIn -= (srcIndex[k] - copySize) * srcPitch[k];
        } else if ((int)batchedDims <= j) {
          offsetIndices -= (indicesIndex[j - batchedDims] - 1) *
                           indicesPitch[j - batchedDims];
          int64_t index_next = indexT[offsetIndices];
          index_next = (index_next < 0) ? index_next + srcIndex[batchedDims] : index_next;
          offsetIn += (index_next - index) * srcPitch[batchedDims];
          index = index_next;
        } else {
          offsetIn -= (srcIndex[j] - 1) * srcPitch[j];
        }
        copySize = 1; // Reset copy size to default
      } else {
        done = true; // The end of the destination tensor has been reached.
      }
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
