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

#ifndef _SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_
#define _SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "Float16.h"
#include "Writer.h" // From include/internal path
#include "Addresser.h" // From include/internal path
#include "Converter.h" // From include/internal path
#include "Operator.h" // From include/internal path
#include "utils.h" // From include/internal path
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

// This version does NOT support Tensors of more than 2 dimensions with padding
template <typename srcType>
inline void fwdLibSparseLengthsWeightedSumInst(LibTensor* outT, LibTensor* in1T,
                                               LibTensor* in2T, LibTensor* in3T,
                                               LibTensor* in4T,
                                               unsigned int pLengthsSize) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* data = in1T->getRawDataPointer<void>();
  void* weight = in2T->getRawDataPointer<void>();
  
  // Addresser<srcType> tOutput(pdst, scale[4], offset[4]);
  Addresser<srcType> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  const Addresser<srcType> tAInput(data, in1T->getScale(), in1T->getOffset());
  // const Addresser<srcType> tWInput(pweights, scale[1], offset[1]);
  const Addresser<srcType> tWInput(weight, in2T->getScale(), in2T->getOffset());
  // long long *indices = (long long *)pindices;
  long long *indices = in3T->getRawDataPointer<long long>();
  // int32_t *lengths = (int32_t *)plengths;
  int32_t *lengths = in4T->getRawDataPointer<int32_t>();

  // unsigned int *dataIndex = (unsigned int *)pdataDims;
  const size_t *dataIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const size_t *dstPitch = outT->strides().data();
  // unsigned int *dataPitch = (unsigned int *)pdataPitches;
  const size_t *dataPitch = in1T->strides().data();
  // unsigned int *weightPitch = (unsigned int *)pweightsPitches;
  const size_t *weightPitch = in2T->strides().data();

  unsigned int pdstDimNum = static_cast<unsigned int>(outT->ndims());

  size_t segments = pLengthsSize;
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += lengths[i];
  }

  size_t totalSize = 1;
  for (size_t i = 0; i < pdstDimNum; i++) {
    totalSize *= dataIndex[i];
  }
  size_t lineSize = totalSize / dataIndex[0];

  // Output tensor should be zero at the begin
  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    // NOTE : Not C++ compliant?  Fails with clang.
    // float tmp[lineSize] = { 0.0f };
    float tmp[lineSize];
    for (size_t j = 0; j < lineSize; j++)
      tmp[j] = 0.0f;
    for (size_t j = 0, e = lengths[i]; j < e; j++) {
      float weight = tWInput[curIdx * weightPitch[0]];
      size_t offsetIn = indices[curIdx] * dataPitch[0];
      for (size_t k = 0; k < lineSize; k++) {
        tmp[k] += tAInput[offsetIn] * weight;
        offsetIn++;
      }
      curIdx++;
    }
    size_t offsetOut = i * dstPitch[0];
    for (size_t k = 0; k < lineSize; k++) {
      tOutput[offsetOut] = tmp[k];
      offsetOut++;
    }
  }
}

// This version DOES support Tensors of more than 2 dimensions with padding
template <typename srcType>
inline void fwdLibSparseLengthsWeightedSumInstThreaded(LibTensor* outT,
                                                       LibTensor* in1T,
                                                       LibTensor* in2T,
                                                       LibTensor* in3T,
                                                       LibTensor* in4T,
                                                       unsigned int pLengthsSize,
                                                       uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* data = in1T->getRawDataPointer<void>();
  void* weight = in2T->getRawDataPointer<void>();
 
  // Addresser<srcType> tOutput(pdst, scale[4], offset[4]);
  Addresser<srcType> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  const Addresser<srcType> tAInput(data, in1T->getScale(), in2T->getOffset());
  // const Addresser<srcType> tWInput(pweights, scale[1], offset[1]);
  const Addresser<srcType> tWInput(weight, in2T->getScale(), in2T->getOffset());
  // long long *indices = (long long *)pindices;
  long long *indices = in3T->getRawDataPointer<long long>();
  // int32_t *lengths = (int32_t *)plengths;
  int32_t *lengths = in4T->getRawDataPointer<int32_t>();

  // unsigned int *dstIndex = (unsigned int *)pdstDims;
  const size_t *dstIndex = outT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const size_t *dstPitch = outT->strides().data();
  // unsigned int *dataPitch = (unsigned int *)pdataPitches;
  const size_t *dataPitch = in1T->strides().data();
  // unsigned int *weightPitch = (unsigned int *)pweightsPitches;
  const size_t *weightPitch = in2T->strides().data();

  unsigned int pdstDimNum = static_cast<unsigned int>(outT->ndims());
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  size_t segments = pLengthsSize;
  size_t ranges[segments];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    ranges[i] = totalLength;
    totalLength += lengths[i];
  }

  unsigned int coord[pdstDimNum];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, pdstDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++)
    offsetOut += coord[i] * dstPitch[i];
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    size_t segment_begin = ranges[coord[0]];
    size_t segment_end = segment_begin + lengths[coord[0]];

    size_t offsetIn = 0;
    for (unsigned int i = 1; i < pdstDimNum; i++)
      offsetIn += coord[i] * dataPitch[i];

    float res = 0;
    for (size_t k = segment_begin; k < segment_end; k++) {
      res += tAInput[indices[k] * dataPitch[0] + offsetIn] *
             (float)tWInput[k * weightPitch[0]];
    }

    tOutput[offsetOut] = res;
    done = getOffsets(pdstDimNum, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

} // namespace dnn_lib

} // namespace inlining

#endif // _SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_
