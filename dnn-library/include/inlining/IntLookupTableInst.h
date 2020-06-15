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

#ifndef _INT_LOOKUP_TABLE_INST_H_
#define _INT_LOOKUP_TABLE_INST_H_

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

inline void fwdLibIntLookupTableInst(LibTensor* outT, LibTensor* in1T,
                                     LibTensor* in2T) {
  assert(outT->getElementType() == Int8QTy &&
         in1T->getElementType() == Int8QTy &&
         in2T->getElementType() == Int8QTy);
  
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */

  // int8_t *ptrDstT = (int8_t *)dstT;
  int8_t *ptrDstT = outT->getRawDataPointer<int8_t>();
  // int8_t *ptrSrcT1 = (int8_t *)src1T;
  int8_t *ptrSrcT1 = in1T->getRawDataPointer<int8_t>();
  // int8_t *ptrSrcT2 = (int8_t *)src2T;
  int8_t *ptrSrcT2 = in2T->getRawDataPointer<int8_t>();
  
  // unsigned int *src1Index = (unsigned int *)src1Dims;
  const dim_t* src1Index =  in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t* dstPitch = outT->strides().data();
  // unsigned int *src1Pitch = (unsigned int *)src1Pitches;
  const dim_t* src1Pitch = in1T->strides().data();

  unsigned int dstDimNum = static_cast<unsigned int>(outT->ndims());
  
  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc1Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < dstDimNum; i++) {
    eDims[i] = src1Index[i];
    eDstPitch[i] = dstPitch[i];
    eSrc1Pitch[i] = src1Pitch[i];
  }

  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              int val = ptrSrcT1[x * eSrc1Pitch[0] + y * eSrc1Pitch[1] +
                                 z * eSrc1Pitch[2] + w * eSrc1Pitch[3] +
                                 q * eSrc1Pitch[4] + r * eSrc1Pitch[5]];
              ptrDstT[x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                      w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5]] =
                  ptrSrcT2[val + 128];
            }
          }
        }
      }
    }
  }
}

inline void fwdLibIntLookupTableInstThreaded(LibTensor* outT,
                                             LibTensor* in1T,
                                             LibTensor* in2T,
                                             uint64_t flags) {
  assert(outT->getElementType() == Int8QTy &&
         in1T->getElementType() == Int8QTy &&
         in2T->getElementType() == Int8QTy);
  
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  void *dst = outT->getRawDataPointer<void>();
  
  // int8_t *ptrDstT = (int8_t *)dstT;
  int8_t *ptrDstT = outT->getRawDataPointer<int8_t>();
  // int8_t *ptrSrcT1 = (int8_t *)src1T;
  int8_t *ptrSrcT1 = in1T->getRawDataPointer<int8_t>();
  // int8_t *ptrSrcT2 = (int8_t *)src2T;
  int8_t *ptrSrcT2 = in2T->getRawDataPointer<int8_t>();
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t* dstIndex =  outT->dims().data();
  // unsigned int *src1Index = (unsigned int *)src1Dims;
  const dim_t* src1Index =  in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t* dstPitch = outT->strides().data();
  // unsigned int *src1Pitch = (unsigned int *)src1Pitches;
  const dim_t* src1Pitch = in1T->strides().data();

  unsigned int dstDimNum = static_cast<unsigned int>(outT->ndims());

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  getCachelinePartition(sizeof(int8_t), numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[dstDimNum];
  unsigned int k;
  
  /* overloading while sw-2400 and sw-2429 are WIP */
  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex, k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += src1Pitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;

  bool done = false;
  while (!done && (offsetOut < posMax)) {
    ptrDstT[offsetOut] = ptrSrcT2[ptrSrcT1[offsetIn] + 128];
    done = getOffsets(dstDimNum, coord, offsetIn, offsetOut, src1Index,
                      src1Pitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * sizeof(int8_t) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + sizeof(int8_t)*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _INT_LOOKUP_TABLE_INST_H_
