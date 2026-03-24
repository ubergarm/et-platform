/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _LENGTHS_SUM_INST_H
#define _LENGTHS_SUM_INST_H

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

template <ElemKind elK>
INLINE_ATTR void fwdLibLengthsSumInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,
                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* outT --> dst  in1T--> src in2T--> index*/
  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer();
  void* src = in1T->getRawDataPointer();

  Addresser<elK> tOutput(dst, outT->getScale(), outT->getOffset());
  const Addresser<elK> tTmp(dst, outT->getScale(), outT->getOffset());
  const Addresser<elK> tAInput(src, in1T->getScale(), in1T->getOffset());
  auto lengths = in2T->getRawDataPointer<int32_t>();

  const dim_t *dstIndex = outT->dims().data();
  const dim_t *dataIndex = in1T->dims().data();

  const dim_t *dstPitch = outT->strides().data();
  const dim_t *dataPitch = in1T->strides().data();

  size_t pdataDimNum = static_cast<unsigned int>(in1T->ndims());
  size_t numElemsDst = dstPitch[0] * dstIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address and the number of positions that it must work on (maxRead).
  size_t initialAddr, maxRead;
  getCachelinePartition(sizeof(srcType), numElemsDst, initialAddr, maxRead, minionId, activeMinions, dst);
  if (maxRead == 0) return;

  // We move the initialAddr to the next non-padding position

  dim_t k;                    // Amount of non-zero coordinates
  dim_array_t coordIn = {0};  // Vector of coordinates
  dim_array_t coordOut = {0}; // Vector of coordinates

  getNonPaddingCoordinates(coordOut, initialAddr, pdataDimNum, dstPitch, dstIndex, k);
  coordIn[0] = 0;
  for (dim_t i = 1; i < pdataDimNum; i++) {
    coordIn[i] = coordOut [i];
  }
  for (dim_t l = 0; l < coordOut[0]; l++)
    coordIn[0] += lengths[l];

  size_t offsetIn = 0;
  size_t offsetOut = 0;
  for (dim_t j = 0; j < k; j++) {
    offsetIn += dataPitch[j]*coordIn[j];
    offsetOut += dstPitch[j]*coordOut[j];
  }
  size_t offsetIn0 = offsetIn;
  size_t offsetOut0 = offsetOut;

  dim_array_t coordIn0 = {0};
  dim_array_t coordOut0 = {0};

  for (dim_t i = 0; i < pdataDimNum; i++) {
    coordIn0[i] = coordIn[i];
    coordOut0[i] = coordOut[i];
  }

  size_t posMax = maxRead + initialAddr;
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
      }
      offsetIn = offsetIn0 + (posIn + 1) * dataPitch[0];
      offsetOut = offsetOut0;

      for (dim_t i = 0; i < pdataDimNum; i++) {
        coordIn[i] = coordIn0[i];
        coordOut[i] = coordOut0[i];
      }
      coordIn[0] += (posIn + 1);
      endmatrix = false;
    }
    if (done) {
      break;
    }
    offsetIn = 0;
    offsetIn0 = 0;
    offsetOut = 0;
    offsetOut0 = 0;
    for (dim_t i = 1; i < pdataDimNum; i++) {
      coordIn[i] = coordIn0[i] = 0;
      coordOut[i] = coordOut0[i] = 0;
    }
    for (dim_t j = 0; j <= coordOut0[0]; j++) {
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
  size_t clperminion = (maxRead * sizeof(srcType) + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + sizeof(srcType)*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _LENGTHS_SUM_INST_H
