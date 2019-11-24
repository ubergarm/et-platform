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

void dnn_lib::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  float *tOutput = (float *)pdst;
  uint8_t *tAInput = (uint8_t *)pdata;
  float *tScale = (float *)pscale;
  float *tOffset = (float *)poffset;
  float *tWInput = (float *)pweights;
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  size_t segments = pLengthsSize;
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += lengths[i];
  }
  // assert(totalLength == weightIndex[0] && "sum(Lengths) must be equal to
  // len(Indices)");

  size_t totalSize = 1;
  for (size_t i = 0; i < pdstDimNum; i++) {
    totalSize *= dataIndex[i];
  }
  size_t lineSize = totalSize / dataIndex[0];

  // Output tensor should be zero at the begin
  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    for (size_t j = 0, e = lengths[i]; j < e; j++) {
      const float weight = tWInput[curIdx * weightPitch[0]];
      const size_t rowIdx = indices[curIdx];
      const float scale = tScale[rowIdx];
      const float offset = tOffset[rowIdx];
      size_t offsetIn = rowIdx * dataPitch[0];
      size_t offsetOut = i * dstPitch[0];
      curIdx++;
      for (size_t k = 0; k < lineSize; k++) {

        float d = dequantizeWithFloatOffset(tAInput[offsetIn], scale, offset);
        tOutput[offsetOut] += d * weight;
        offsetOut++;
        offsetIn++;
      }
    }
  }
}


/*
void dnn_lib::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags) {
  float *tOutput = (float *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  float *tScale = (float *)pscale;
  float *tOffset = (float *)poffset;
  float *tWInput = (float *)pweights;
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];
  unsigned int initialAddr, maxRead;
  getCachelinePartition(sizeof(float), numElemsDst, initialAddr, maxRead,
activeMinions);

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
  for (int i = 0; i < k; i++) offsetOut += coord[i]*dstPitch[i];
  if (offsetOut >= numElemsDst) return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  while(!done) {
    size_t segment_begin = ranges[coord[0]];
    size_t segment_end = segment_begin + lengths[coord[0]];

    size_t offsetIn = 0;
    for (int i = 1; i < pdstDimNum; i++) offsetIn += coord[i] * dataPitch[i];

    float res = 0;
    for (size_t k = segment_begin; k < segment_end; k++) {
      size_t idx = indices[k];
      float d = dequantizeWithFloatOffset(tAInput[indices[k]*dataPitch[0] +
offsetIn], tScale[k], tOffset[k]); res += d * (float)tWInput[k *
weightPitch[0]];
    }

    tOutput[offsetOut] = res;

    done = getOffsets(pdstDimNum, coord, offsetOut, dstIndex, dstPitch);
    if (offsetOut >= posMax) break;
  }
}
*/

void dnn_lib::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  float *tOutput = (float *)pdst;
  uint8_t *tAInput = (uint8_t *)pdata;
  float *tScale = (float *)pscale;
  float *tOffset = (float *)poffset;
  float *tWInput = (float *)pweights;
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  size_t segments = pLengthsSize;
  size_t ranges[segments];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    ranges[i] = totalLength;
    totalLength += lengths[i];
  }
  // assert(totalLength == weightIndex[0] && "sum(Lengths) must be equal to
  // len(Indices)");

  size_t lineSize = 1;
  for (size_t i = 1; i < pdstDimNum; i++)
    lineSize *= dataIndex[i];

  unsigned int numElemsDst = dstPitch[0] * segments;
  unsigned int cll = 64 / sizeof(float);
  unsigned int rowsperminion = (cll - 1) / dstPitch[0] + 1;
  unsigned int total_rows = rowsperminion * activeMinions;
  for (unsigned int i = total_rows; i < segments; i += activeMinions)
    rowsperminion++;
  unsigned int row_begin = minionId * rowsperminion;
  if (row_begin >= segments)
    return;
  unsigned int row_end = row_begin + rowsperminion;

  size_t curIdx = ranges[row_begin];
  for (size_t i = row_begin; i < row_end; i++) {
    for (size_t j = 0, e = lengths[i]; j < e; j++) {
      const float weight = tWInput[curIdx * weightPitch[0]];
      const size_t rowIdx = indices[curIdx];
      const float scale = tScale[rowIdx];
      const float offset = tOffset[rowIdx];
      size_t offsetIn = rowIdx * dataPitch[0];
      size_t offsetOut = i * dstPitch[0];
      curIdx++;
      for (size_t k = 0; k < lineSize; k++) {

        float d = dequantizeWithFloatOffset(tAInput[offsetIn], scale, offset);
        tOutput[offsetOut] += d * weight;
        offsetOut++;
        offsetIn++;
      }
    }
  }
}

