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

#ifndef _BATCH_ONE_HOST_INST_H_
#define _BATCH_ONE_HOST_INST_H_

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

template <ElemKind elK>
INLINE_ATTR void fwdLibBatchOneHotInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                                       uint64_t flags, const uint32_t minionOffset = 0,
                                       const uint32_t assignedMinions = 0) {

  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer();
  void* dataT = in1T->getRawDataPointer();
  void* valuesT = in3T->getRawDataPointer();

  // Addresser<elK> tOutput(pdst, scale[2], offset[2]);
  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tAInput(pdata, scale[0], offset[0]);
  const Addresser<elK> tAInput(dataT, in1T->getScale(), in1T->getOffset());
  // const Addresser<elK> tValues(pvalues, scale[1], offset[1]);
  const Addresser<elK> tValues(valuesT, in3T->getScale(), in3T->getOffset());
  // int32_t *lengths = (int32_t *)plengths;
  uint32_t* lengths = in2T->getRawDataPointer<uint32_t>();

  // unsigned int *dstIndex = (unsigned int *)pdstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dataIndex = (unsigned int *)pdataDims;
  const dim_t *dataIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch  = outT->strides().data();
  // unsigned int *dataPitch = (unsigned int *)pdataPitches;
  const dim_t *dataPitch = in1T->strides().data();
 
  auto batchSize = dataIndex[0];
  auto featureCnt = dataIndex[1];

  auto numElemsDst = batchSize * dstPitch[0]; // Total number of elements in the tensor

  size_t dstAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, dstAddr, maxRead,
                        minionId, activeMinions, dstT);

  if (maxRead == 0) {
    return;
  }

  // We move the initialAddr to the next non-padding position

  dim_t k;                 // Amount of non-zero coordinates
  dim_array_t coord = {0}; // Vector of coordinates

  getNonPaddingCoordinates(coord, dstAddr, 2, dstPitch, dstIndex, k);

  // We get the actual initialAddr, in the input and output.
  dim_t offsetOut = 0;
  for (dim_t j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }
  auto batchId = offsetOut / dstPitch[0];
  auto i = offsetOut - batchId * dstPitch[0];
  auto l = i;
  dim_t featureId = 0;
  while (l >= lengths[featureId]) {
    l -= lengths[featureId];
    featureId++;
    if (featureId >= featureCnt) {
      featureId = 0;
      l = i = 0;
      batchId++;
      break;
    }
  }

  auto posMax = maxRead + dstAddr;

  bool done = (offsetOut >= posMax);
  bool minionEnd = false;
  size_t firstOffset = l;

  while ((not done) and (not minionEnd)) {
    while (batchId < batchSize) {
      size_t offset = i;
      while (featureId < featureCnt) {
        auto curValue = tAInput[batchId * dataPitch[0] + featureId];
        auto curLength = lengths[featureId] - firstOffset;
        firstOffset = 0;
        while (i < offset + curLength) {
          if (curValue == tValues[i]) {
            tOutput[offsetOut] = (float)1;
          } else {
            tOutput[offsetOut] = (float)0;
          }
          offsetOut++;
          if (offsetOut >= posMax) {
            minionEnd = true;
            break;
          }
          i++;
        }
        offset += curLength;
        featureId++;
        if (minionEnd) {
          break;
        }
      }
      offsetOut += (dstPitch[0] - i);
      i = 0;
      featureId = 0;
      batchId++;
      if (offsetOut >= posMax) {
        minionEnd = true;
        break;
      }
    }
    if (batchId == batchSize) {
      done = true;
    }
  }

  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*dstAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _BATCH_ONE_HOST_INST_H_
