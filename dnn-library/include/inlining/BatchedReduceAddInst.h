/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _BATCHED_REDUCE_ADD_INST_H_
#define _BATCHED_REDUCE_ADD_INST_H_

#include "Addresser.h"
#include "Float16.h"
#include "LibTensor.h"
#include "Operator.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <limits>
#include <string.h>
#include <utility>

// Accumulates all of the layers in the batch and produce a tensor
// that has the same dimensions as the input tensor without the first dimension;

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
INLINE_ATTR void fwdLibBatchedReduceAddInst(LibTensor* outT, LibTensor* inT, dim_t axis, uint64_t flags,
                                            const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer();
  void* batchT = inT->getRawDataPointer();
  constexpr bool globalStore = true;
  Addresser<elK, globalStore> tOutput(dstT, outT->getScale(), outT->getOffset());
  const Addresser<elK> tBatch(batchT, inT->getScale(), inT->getOffset());

  const dim_t *dstIndex = outT->dims().data();
  const dim_t *batchIndex = inT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *batchPitch = inT->strides().data();

  const dim_t pbatchDimNum = inT->ndims();
  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;

  // requesting a globalPartition since we use globalStores
  getGlobalPartition(numElemsDst, /*out*/ initialAddr, /*out*/ maxRead, minionId, activeMinions);

  if (maxRead == 0) {
    return;
  }

  dim_t redBatchPitch[pbatchDimNum - 1];
  for (dim_t i = 0; i < pbatchDimNum; i++) {
    if (i < axis) {
      redBatchPitch[i] = batchPitch[i];

    } else if (i > axis) {
      redBatchPitch[i - 1] = batchPitch[i];
    }
  }

  dim_array_t offsets = {0};
  dim_t k;
  getNonPaddingCoordinates(/*out*/ offsets, initialAddr, pbatchDimNum - 1, dstPitch, dstIndex,
                           /*out*/ k);

  size_t offsetOut = 0;
  size_t offsetIn = 0;
  for (size_t j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * offsets[j];
    offsetIn += redBatchPitch[j] * offsets[j];
  }

  auto posMax = maxRead + initialAddr;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    // build an addresser out of a stack var to accumulate sums.
    srcType sumVar;
    constexpr bool sumGlobalStore = false;
    Addresser<elK, sumGlobalStore> sum(&sumVar, outT->getScale(), outT->getOffset());
    sum[0] = tBatch[offsetIn];
    offsetIn += batchPitch[axis];

    for (size_t i = 1; i < batchIndex[axis]; i++) {
      // using std::as_const to use the read [] overload
      sum[0] = std::as_const<Addresser<elK, sumGlobalStore>>(sum)[0] + tBatch[offsetIn];

      offsetIn += batchPitch[axis];
    }
    offsetIn -= batchIndex[axis] * batchPitch[axis];
    // use the global-store addresser just here to store the reduced val
    tOutput[offsetOut] = std::as_const<Addresser<elK, sumGlobalStore>>(sum)[0];

    done = getOffsets(pbatchDimNum - 1, /* inout */ offsets, /*inout */ offsetIn, /*inout */ offsetOut, dstIndex,
                      redBatchPitch, dstPitch);
  }
  // This implementation uses globalStores. no evict code needed.
}

template <>
INLINE_ATTR void fwdLibBatchedReduceAddInst<Int8QTy>(LibTensor* outT, LibTensor* inT, dim_t axis, uint64_t flags,
                                                     const uint32_t minionOffset, const uint32_t assignedMinions) {
  constexpr ElemKind elK = Int8QTy;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) {
    return;
  }

  auto* tOutput = outT->getRawDataPointer<int8_t>();
  auto* tBatch = inT->getRawDataPointer<int8_t>();

  float invScale;
  getReciprocal(outT->getScale(), invScale);
  const dim_t *dstIndex = outT->dims().data();
  const dim_t *batchIndex = inT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *batchPitch = inT->strides().data();

  dim_t pbatchDimNum = inT->ndims();
  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialAddr, maxRead;
  // requesting a globalPartition sice we use globalStores
  getGlobalPartition(numElemsDst, /*out*/ initialAddr, /*out*/ maxRead, minionId, activeMinions);

  if (maxRead == 0) {
    return;
  }

  dim_array_t offsets = {0};
  dim_t k;
  dim_t redBatchPitch[pbatchDimNum - 1];
  for (dim_t i = 0; i < pbatchDimNum; i++) {
    if (i < axis) {
      redBatchPitch[i] = batchPitch[i];

    } else if (i > axis) {
      redBatchPitch[i - 1] = batchPitch[i];
    }
  }

  getNonPaddingCoordinates(/*out*/ offsets, initialAddr, pbatchDimNum - 1, dstPitch, dstIndex,
                           /*out*/ k);
  size_t offsetOut = 0;
  size_t offsetIn = 0;
  for (size_t j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * offsets[j];
    offsetIn += redBatchPitch[j] * offsets[j];
  }

  constexpr size_t dstBytesPerElement = Type::getElementSize(elK);
  constexpr bool aligned = false;
  constexpr bool globalStore = true;

  // Vector instructions working as scalar
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n");

  uint64_t dstConf;
  float dstIndices;
  setupGatherScatterConfig<dstBytesPerElement, aligned>(dstConf, dstIndices);

  auto posMax = maxRead + initialAddr;
  bool done = false;
  while (not done && offsetOut < posMax) {
    float sum = 0.f;
    for (size_t i = 0; i < batchIndex[axis]; i++) {
      sum += static_cast<float>(tBatch[offsetIn] - inT->getOffset());
      offsetIn += batchPitch[axis];
    }
    offsetIn -= batchIndex[axis] * batchPitch[axis];
    sum = sum * inT->getScale() * invScale + static_cast<float>(outT->getOffset()); // quantize
    convertFloatToInt32<RoundingMode::LikeStdRoundAndCast>(sum, sum);

    saturateInt8(sum, sum);
    uintptr_t dstAddr = reinterpret_cast<uintptr_t>(&tOutput[offsetOut]);
    store<dstBytesPerElement, aligned, globalStore>(dstAddr, dstConf, dstIndices, sum);

    done = getOffsets(pbatchDimNum - 1, /*inout*/ offsets, /*inout*/ offsetIn, /*inout*/ offsetOut, dstIndex,
                      redBatchPitch, dstPitch);
  }
  // This implementation uses global stores. no evict code needed.
}

} // namespace inlining

} // namespace dnn_lib

#endif // _BATCHED_REDUCE_ADD_INST_H_
