/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _SPARSE_TO_DENSE_MASK_INST_H
#define _SPARSE_TO_DENSE_MASK_INST_H

#include "Addresser.h"
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

// Assumptions for the SparseToDenseMaskInst threaded version:
// (1) The pmask vector size (pMaskSize) has the same length as the second dimension of the output tensor.
// (2) The dimensions of the pdefault tensor are the ones of a batch of the data tensor.

template <ElemKind elK, size_t N>
INLINE_ATTR void fwdLibSparseToDenseMaskInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                                             LibTensor* in4T, const std::array<size_t, N> mask, uint64_t flags,
                                             const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* out--> dest in2T->val in3T->dft in1T->idx in4T->len*/
  void* pdst = outT->getRawDataPointer();
  void* pdata = in2T->getRawDataPointer();
  void* pdefault = in3T->getRawDataPointer();

  Addresser<elK> tOutput(pdst, outT->getScale(), outT->getOffset());
  const Addresser<elK> tAInput(pdata, in2T->getScale(), in2T->getOffset());
  const Addresser<elK> tDefVInput(pdefault, in3T->getScale(), in3T->getOffset());
  auto indices = in1T->getRawDataPointer<size_t>();
  auto lengths = in4T->getRawDataPointer<int32_t>();

  const dim_t* dstIndex = outT->dims().data();
  const dim_t* dstPitch = outT->strides().data();
  const dim_t* dataPitch = in2T->strides().data();
  const dim_t* defPitch = in3T->strides().data();
  const dim_t* defIndex = in3T->dims().data();
  const dim_t* lenIndex = in4T->dims().data();

  dim_t pdstDimNum = outT->ndims();
  dim_t pdataDimNum = in2T->ndims();

  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, pdst);
  if (maxRead == 0)
    return;

  dim_array_t coordOut = {0};
  dim_t last_non_zero_coord;
  getNonPaddingCoordinates(coordOut, initialAddr, pdstDimNum, dstPitch, dstIndex, last_non_zero_coord);

  size_t offsetOut = 0;
  for (dim_t i = 0; i < last_non_zero_coord; i++) {
    offsetOut += dstPitch[i]*coordOut[i];
  }

  dim_t pdefDimNum = pdataDimNum - 1;

  size_t batchCount;
  size_t semiBatchCount;
  dim_array_t coordIn = {0}; // Coordinates in the default value tensor (or an input batch).

  if (lenIndex[0] > 1) {
    size_t remainder = offsetOut;
    batchCount = remainder / dstPitch[0];
    remainder = remainder % dstPitch[0];
    semiBatchCount = remainder / dstPitch[1];
    remainder = remainder % dstPitch[1];
    for (size_t i = 0; i < pdefDimNum; ++i) {
      coordIn[i] = remainder / dstPitch[i + 2];
      remainder = remainder % dstPitch[i + 2];
    }
  } else {
    batchCount = 0;
    semiBatchCount = offsetOut / dstPitch[0];
    size_t remainder = offsetOut % dstPitch[0];
    for (size_t i = 0; i < pdefDimNum; ++i) {
      coordIn[i] = remainder / dstPitch[i + 1];
      remainder = remainder % dstPitch[i + 1];
    }
  }

  size_t offsetIn = 0;
  size_t offsetDef = 0;

  for (dim_t i = 0; i < pdefDimNum; i++) {
    offsetIn += dataPitch[i + 1] * coordIn[i];
    offsetDef += defPitch[i] * coordIn[i];
  }

  size_t firstIdx = 0;
  for (size_t i = 0; i < batchCount; ++i) {
    firstIdx += lengths[i];
  }

  size_t lastIdx = firstIdx + lengths[batchCount];
  size_t idx = mask[semiBatchCount];
  bool defaultVal = true;
  size_t j;

  for (j = firstIdx; j < lastIdx; j++) {
    if (indices[j] == idx) {
      defaultVal = false;
      break;
    }
  }

  size_t posMax = maxRead + initialAddr;
  bool done = false;
  bool doneIn = false;

  while (not done and offsetOut < posMax) {
    srcType value;
    if (defaultVal) {
      value = static_cast<srcType>(tDefVInput[offsetDef]);
    } else {
      value = static_cast<srcType>(tAInput[j * dataPitch[0] + offsetIn]);
    }
    tOutput[offsetOut] = value;

    done = getOffsets(pdstDimNum, coordOut, offsetOut, dstIndex, dstPitch);

    if (done or not(offsetOut < posMax)) {
      break;
    }

    doneIn = getOffsets(pdefDimNum, coordIn, offsetIn, offsetDef, defIndex, dataPitch + 1, defPitch);

    if (doneIn) {
      doneIn = false;
      offsetIn = 0;
      offsetDef = 0;
      for (dim_t i = 0; i < pdefDimNum; ++i) {
        coordIn[i] = 0;
      }
      ++semiBatchCount;
      if (semiBatchCount == mask.size()) { // Assumption (1): pMaskSize = dstIndex[1].
        semiBatchCount = 0;
        ++batchCount;
        firstIdx = lastIdx;
        lastIdx += lengths[batchCount];
      }
      idx = mask[semiBatchCount];
      defaultVal = true;
      for (j = firstIdx; j < lastIdx; j++) {
        if (indices[j] == idx) {
          defaultVal = false;
          break;
        }
      }
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _SPARSE_TO_DENSE_MASK_INST_H
