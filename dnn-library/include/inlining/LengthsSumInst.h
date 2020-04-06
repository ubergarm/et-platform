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

#ifndef _LENGTHS_SUM_INST_H
#define _LENGTHS_SUM_INST_H

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
inline void fwdLibLengthsSumInst(void *pdst, void *pdstDims,
                                   void *pdstPitches, void *pdata,
                                   void *pdataDims, void *pdataPitches,
                                   unsigned int pdataDimNum, void *plengths,
                                   unsigned int pLengthsSize, const float *scale,
                                   const int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId > 0)
    return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tTmp(pdst, scale[2], offset[2]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eDataPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < pdataDimNum; i++) {
    eDims[i] = dataIndex[i];
    eDstPitch[i] = dstPitch[i];
    eDataPitch[i] = dataPitch[i];
  }

  uint64_t addrSrc, addrDst;
  // Initialize to zero output tensor
  for (size_t i = 0; i < pLengthsSize; i++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              addrDst = i * eDstPitch[0] + y * eDstPitch[1] +
                        z * eDstPitch[2] + w * eDstPitch[3] +
                        q * eDstPitch[4] + r * eDstPitch[5];
              tOutput[addrDst] = 0;
            }
          }
        }
      }
    }
  }

  // Global index inside data
  size_t posIn = 0;
  for (size_t i = 0; i < pLengthsSize; i++) {
    for (int32_t j = 0, e = lengths[i]; j < e; j++, posIn++) {
      // Sum elements across batch dimension
      for (size_t y = 0; y < eDims[1]; y++) {
        for (size_t z = 0; z < eDims[2]; z++) {
          for (size_t w = 0; w < eDims[3]; w++) {
            for (size_t q = 0; q < eDims[4]; q++) {
              for (size_t r = 0; r < eDims[5]; r++) {
                addrDst = i * eDstPitch[0] + y * eDstPitch[1] +
                          z * eDstPitch[2] + w * eDstPitch[3] +
                          q * eDstPitch[4] + r * eDstPitch[5];
                addrSrc = posIn * eDataPitch[0] + y * eDataPitch[1] +
                          z * eDataPitch[2] + w * eDataPitch[3] +
                          q * eDataPitch[4] + r * eDataPitch[5];
                tOutput[addrDst] = tTmp[addrDst] + tAInput[addrSrc];
              }
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
inline void fwdLibLengthsSumInstThreaded(void *pdst, void *pdstDims,
                                           void *pdstPitches, void *pdata,
                                           void *pdataDims, void *pdataPitches,
                                           unsigned int pdataDimNum, void *plengths,
                                           unsigned int pLengthsSize, const float *scale,
                                           const int32_t *offset, uint64_t flags) {

  unsigned int minion_id = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minion_id >= activeMinions) return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tTmp(pdst, scale[2], offset[2]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address and the number of positions that it must work on (maxRead).
  unsigned int initialAddr, maxRead;
  getCachelinePartition(sizeof(srcType), numElemsDst, initialAddr, maxRead, minion_id, activeMinions);
  if (maxRead == 0) return;

  // We move the initialAddr to the next non-padding position

  unsigned int k; // Amount of non-zero coordinates
  unsigned int coordIn[pdataDimNum];  // Vector of coordinates
  unsigned int coordOut[pdataDimNum]; // Vector of coordinates

  getNonPaddingCoordinates(coordOut, initialAddr, pdataDimNum, dstPitch, dstIndex, k);
  coordIn[0] = 0;
  for (unsigned int i = 1; i < pdataDimNum; i++) {
    coordIn[i] = coordOut [i];
  }
  for (unsigned int l = 0; l < coordOut[0]; l++) coordIn[0] += lengths[l];

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += dataPitch[j]*coordIn[j];
    offsetOut += dstPitch[j]*coordOut[j];
  }
  unsigned int offsetIn0 = offsetIn;
  unsigned int offsetOut0 = offsetOut;
  unsigned int offsetOutmax;

  unsigned int coordIn0[pdataDimNum];
  unsigned int coordOut0[pdataDimNum];

  for (unsigned int i = 0; i < pdataDimNum; i++) {
    coordIn0[i] = coordIn[i];
    coordOut0[i] = coordOut[i];
  }

  unsigned int posMax = maxRead + initialAddr;
  // initialize output tensor
  for (size_t elem = initialAddr; elem < posMax*sizeof(srcType); elem++)
    tOutput[elem] = 0;

  // In each iteration we copy a position and switch to the next one, until completion.
  bool endmatrix = false;
  bool done = false;
  size_t posIn = 0;
  while (!done) {
    for (size_t posIn = 0; posIn < lengths[coordOut[0]]; posIn++) {
      while (!endmatrix && (offsetOut < posMax)) {
        tOutput[offsetOut] = tAInput[offsetIn] + tTmp[offsetOut];
        for (size_t j = pdataDimNum - 1; j > 0; j--) {
          if (coordIn[j] != (dataIndex[j] - 1)) {
            offsetIn += dataPitch[j];
            offsetOut += dstPitch[j];
            coordIn[j]++;
            coordOut[j]++;
            break;
          }
          else if (j != 1) {
            offsetIn -= (dataIndex[j] - 1) * dataPitch[j];
            offsetOut -= (dstIndex[j] - 1) * dstPitch[j];
            coordIn[j] = 0;
            coordOut[j] = 0;
          }
          else {
            if (coordOut[0] == dstIndex[0] - 1) {
              done = true;
             // tOutput[offsetOut] = -2;
            }
            endmatrix = true;
            break;
          }
        }
      }
      if (offsetOut >= posMax) {
        done = true;
      //  tOutput[offsetOut] = offsetOut0;
      //  tOutput[offsetOut + 1] = offsetOut;
      }
      offsetIn = offsetIn0 + (posIn + 1) * dataPitch[0];
      offsetOut = offsetOut0;

      for (unsigned int i = 0; i < pdataDimNum; i++) {
        coordIn[i] = coordIn0[i];
        coordOut[i] = coordOut0[i];
      }
      coordIn[0] += (posIn + 1);
      endmatrix = false;
    }
    if (done) {
     // tOutput[offsetOut] = -10;
      break;
    }
    offsetIn = 0;
    offsetIn0 = 0;
    offsetOut = 0;
    offsetOut0 = 0;
    for (unsigned int i = 1; i < pdataDimNum; i++) {
      coordIn[i] = coordIn0[i] = 0;
      coordOut[i] = coordOut0[i] = 0;
    }
    for (int j = 0; j <= coordOut0[0]; j++) {
      offsetIn += dataPitch[0] * lengths[j];
      offsetIn0 += dataPitch[0] * lengths[j];
      offsetOut += dstPitch[0];
      offsetOut0 += dstPitch[0];
    }
    coordIn[0] += lengths[coordOut0[0]];
    coordIn0[0] += lengths[coordOut0[0]];
    coordOut[0]++;
    coordOut0[0]++;
  }

  if (!DO_EVICTS) return;
  unsigned int clperminion = maxRead*sizeof(srcType)/64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + sizeof(srcType)*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _LENGTHS_SUM_INST_H
