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
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
inline void fwdLibLengthsSumInst(LibTensor* outT, LibTensor* in1T,
                                 LibTensor* in2T, uint64_t flags,
                                 const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* outT --> dst  in1T--> src in2T--> index*/
  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* src = in1T->getRawDataPointer<void>();

  // Addresser<elK> tOutput(pdst, scale[2], offset[2]);
  Addresser<elK> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tTmp(pdst, scale[2], offset[2]);
  const Addresser<elK> tTmp(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tAInput(pdata, scale[0], offset[0]);
  const Addresser<elK> tAInput(src, in1T->getScale(), in1T->getOffset());
  // int32_t *lengths = (int32_t *)plengths;
  int32_t *lengths = in2T->getRawDataPointer<int32_t>();
  
  // unsigned int *dstIndex = (unsigned int *)pdstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dataIndex = (unsigned int *)pdataDims;
  const dim_t *dataIndex = in1T->dims().data();

  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *dataPitch = (unsigned int *)pdataPitches;
  const dim_t *dataPitch = in1T->strides().data();

  unsigned int pdataDimNum = static_cast<unsigned int>(in1T->ndims());
    
  unsigned int numElemsDst = dstPitch[0]*dstIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address and the number of positions that it must work on (maxRead).
  unsigned int initialAddr, maxRead;
  getCachelinePartition(sizeof(srcType), numElemsDst, initialAddr, maxRead, minionId, activeMinions, dst);
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

  unsigned int coordIn0[pdataDimNum];
  unsigned int coordOut0[pdataDimNum];

  for (unsigned int i = 0; i < pdataDimNum; i++) {
    coordIn0[i] = coordIn[i];
    coordOut0[i] = coordOut[i];
  }

  unsigned int posMax = maxRead + initialAddr;
  // initialize output tensor
  for (size_t elem = initialAddr; elem < posMax; elem++)
    tOutput[elem] = 0;

  // In each iteration we copy a position and switch to the next one, until completion.
  bool endmatrix = false;
  bool done = (offsetOut >= posMax);
  while (!done) {
    for (int32_t posIn = 0; posIn < lengths[coordOut[0]]; posIn++) {
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
    for (unsigned int j = 0; j <= coordOut0[0]; j++) {
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
  unsigned int clperminion = (maxRead * sizeof(srcType) + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + sizeof(srcType)*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _LENGTHS_SUM_INST_H
