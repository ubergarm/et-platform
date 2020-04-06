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

namespace dnn_lib {

namespace inlining {

template <typename srcType>
inline void fwdLibBatchOneHotInst(void *pdst, void *pdstDims,
                                    void *pdstPitches, void *pdata,
                                    void *pdataDims, void *pdataPitches,
                                    void *pvalues, void *pvaluesDims,
                                    void *pvaluesPitches, void *plengths,
                                    const float *scale, const int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  const Addresser<srcType> tValues(pvalues, scale[1], offset[1]);
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *valuesIndex = (unsigned int *)pvaluesDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *valuesPitch = (unsigned int *)pvaluesPitches;

  auto batchSize = dataIndex[0];
  auto featureCnt = dataIndex[1];

  for (size_t batchId = 0; batchId < batchSize; batchId++) {
    size_t offset = 0;
    for (size_t featureId = 0; featureId < featureCnt; featureId++) {
      auto curValue = tAInput[batchId * dataPitch[0] + featureId];
      auto curLength = lengths[featureId];
      for (size_t i = offset, e = offset + curLength; i != e; i++) {
        int64_t dstAddr = batchId * dstPitch[0] + i;
        if (curValue == tValues[i]) {
          tOutput[dstAddr] = (float)1;
        } else {
          tOutput[dstAddr] = (float)0;
        }
      }
      offset += curLength;
    }
    // assert(offset == dstIndex[1] && "Sum of Lengths must be equal to size of
    // Values");
  }
}

template <typename srcType>
inline void fwdLibBatchOneHotInstThreaded(void *pdst, void *pdstDims,
                                            void *pdstPitches, void *pdata,
                                            void *pdataDims, void *pdataPitches,
                                            void *pvalues, void *pvaluesDims,
                                            void *pvaluesPitches, void *plengths,
                                            const float *scale, const int32_t *offset, uint64_t flags) {


  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  const Addresser<srcType> tValues(pvalues, scale[1], offset[1]);
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *valuesIndex = (unsigned int *)pvaluesDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *valuesPitch = (unsigned int *)pvaluesPitches;

  auto batchSize = dataIndex[0];
  auto featureCnt = dataIndex[1];

  unsigned int numElemsDst = batchSize * dstPitch[0]; // Total number of elements in the tensor



  unsigned int dstAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, dstAddr, maxRead,
                        minionId, activeMinions);

  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position

  unsigned int k;          // Amount of non-zero coordinates
  unsigned int coord[2]; // Vector of coordinates



  getNonPaddingCoordinates(coord, dstAddr, 2, dstPitch, dstIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int batchId = offsetOut/dstPitch[0];
  unsigned int i = offsetOut - batchId * dstPitch[0];
  unsigned int l = i;
  unsigned int featureId = 0;
  while (l >= lengths[featureId]) {
    l -= lengths[featureId];
    featureId++;
    if (featureId >= featureCnt)  {
      featureId = 0;
      l = i = 0;
      batchId++;
      break;
    }
  }


  unsigned int posMax = maxRead + offsetOut;


  bool done = false;
  bool minionEnd = false;

  while (!done && !minionEnd) {
    while (batchId < batchSize){
      size_t offset = i;
      while (featureId < featureCnt){
        auto curValue = tAInput[batchId * dataPitch[0] + featureId];
        auto curLength = lengths[featureId];
        while (i < offset + curLength) {
          if (curValue == tValues[i]) {
            tOutput[offsetOut] = (float)1;
          } else {
            tOutput[offsetOut] = (float)0;
          }
          offsetOut++;
          if (offsetOut > posMax) {
            minionEnd = true;
            break;
          }
          i++;
        }
        offset += curLength;
        featureId++;
        if (minionEnd == true)
          break;
      }
      i = 0;
      featureId = 0;
      batchId++;
      offsetOut += dstPitch[0];
      if (offsetOut > posMax) {
       minionEnd = true;
      }
      if (minionEnd == true)
        break;
    }
    if (batchId == batchSize)
      done = true;
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*dstAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _BATCH_ONE_HOST_INST_H_
