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

template <typename srcType>
void dnn_lib::fwdLibScatterAssignInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *indexT,
                                      void *indicesDims, void *pindicesPitches,
                                      void *slicesT, void *slicesDims,
                                      void *slicesPitches, float *scale,
                                      int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> tSlices(slicesT, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);

  long long *tIndices = (long long *)indexT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *indicesIndex = (unsigned int *)indicesDims;
  unsigned int *slicesIndex = (unsigned int *)slicesDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *indicesPitch = (unsigned int *)pindicesPitches;
  unsigned int *slicesPitch = (unsigned int *)slicesPitches;

  // For each slice (small fragment) that we copy from the source memory:
  uint64_t n = slicesPitch[0];
  auto val = tSlices[0];
  /*for (int i = 0; i < indicesIndex[0]; i++){
   val =  (int)tIndices[i];

   tOutput[i] = val;
  }
  return; */
  for (int j = 0; j < indicesIndex[0]; j++) {
    // Reads index [j]
    long long index = tIndices[j];
    // std::copy(&tSlices[j*slicesPitch[0]], &tSlices[j*slicesPitch[0]] +
    // slicesPitch[0], &tOutput[index*dstPitch[0]]);
    uint64_t srcAddr = j * n;
    uint64_t dstAddr = index * dstPitch[0];
    // perform the copy
    for (uint64_t i = 0; i < n; i++) {
      val = tSlices[srcAddr + i];
      tOutput[dstAddr + i] = val;
    }
  }
}


template <typename srcType>
void dnn_lib::fwdLibScatterAssignInstThreaded(void *dstT, void *dstDims,
                                              void *dstPitches, unsigned int dstDimNum, void *indexT,
                                              void *indicesDims, void *pindicesPitches,
                                              void *slicesT, void *slicesDims,
                                              void *slicesPitches, float *scale,
                                              int32_t *offset, uint64_t flags) {


  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> tSlices(slicesT, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);

  long long *tIndices = (long long *)indexT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *indicesIndex = (unsigned int *)indicesDims;
  unsigned int *slicesIndex = (unsigned int *)slicesDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *indicesPitch = (unsigned int *)pindicesPitches;
  unsigned int *slicesPitch = (unsigned int *)slicesPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[dstDimNum];
  for (unsigned int i = 0; i < dstDimNum; i++)
    coord[i] = 0;
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetIn = 0; // Doesn't include slicesPitch[0]. offsetIn doesn't
                             // have the conventional meaning
  unsigned int offsetOut = dstPitch[0] * coord[0];

  unsigned int slicesPitch_0 = slicesPitch[0];
  slicesPitch[0] = 0;

  for (unsigned int j = 1; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
    offsetIn += slicesPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  bool change = true;
  int offset0 = 0;
  while (!done && (offsetOut < posMax)) {
    if (change){
      offset0 = indicesIndex[0] - 1;
      while(offset0 >= 0){
        if (tIndices[offset0] == coord[0]) break;
        offset0--;
      }
      change = false;
    }
    if (offset0 >= 0) tOutput[offsetOut] = tSlices[offsetIn + offset0*slicesPitch_0];


    for (int j = dstDimNum - 1; j >= 0; j--) {
      if (coord[j] != (dstIndex[j] - 1)) {
        offsetOut += dstPitch[j];
        offsetIn += slicesPitch[j];
        coord[j]++;
        if (j == 0) change = true;
        break;
      } else if (j != 0) {
        offsetOut -= (dstIndex[j] - 1) * dstPitch[j];
        offsetIn -= (slicesIndex[j] - 1) * slicesPitch[j];
        coord[j] = 0;
      } else {
        done = true;
        break;
        }
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

GEN_INSTANCES_OP(template, fwdLibScatterAssignInst, void *dstT, void *dstDims,
                                void *dstPitches, void *indexT, void *indicesDims, void *pindicesPitches,
                                void *slicesT, void *slicesDims, void *slicesPitches ,
                                float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibScatterAssignInstThreaded, void *dstT, void *dstDims,
                                void *dstPitches, unsigned int dstDimNum, void *indexT, void *indicesDims, void *pindicesPitches,
                                void *slicesT, void *slicesDims, void *slicesPitches ,
                                float *scale, int32_t *offset, uint64_t flags);
