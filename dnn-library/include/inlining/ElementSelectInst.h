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

#ifndef _ELEMENT_SELECT_INST_H_
#define _ELEMENT_SELECT_INST_H_

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

template <typename srcType>
inline void fwdLibElementSelectInst(LibTensor* outT, LibTensor* condT,
                                    LibTensor* in1T, LibTensor* in2T) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer<void>();
  void* srcT1 = in1T->getRawDataPointer<void>();
  void* srcT2 = in2T->getRawDataPointer<void>();
  // bool *ptrCondT = (bool *)condT;
  bool* ptrCondT = condT->getRawDataPointer<bool>();
  
  // Addresser<srcType> ptrDstT(dstT, scale[3], offset[3]);
  Addresser<srcType> ptrDstT(dstT, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> ptrSrcT1(srcT1, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT1(srcT1, in1T->getScale(), in1T->getOffset());
  // const Addresser<srcType> ptrSrcT2(srcT2, scale[2], offset[2]);
  const Addresser<srcType> ptrSrcT2(srcT2, in2T->getScale(), in2T->getOffset());

  // unsigned int *srcIndex = (unsigned int *)srcDims;
  const dim_t *srcIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *src1Pitch = (unsigned int *)src1Pitches;
  const dim_t *src1Pitch = in1T->strides().data();
  // unsigned int *src2Pitch = (unsigned int *)src2Pitches;
  const dim_t *src2Pitch = in2T->strides().data();
  // unsigned int *condPitch = (unsigned int *)condPitches;
  const dim_t *condPitch = condT->strides().data();
 
  unsigned int srcDimNum = static_cast<unsigned int>(in1T->ndims());
  
  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc1Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc2Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eCondPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrc1Pitch[i] = src1Pitch[i];
    eSrc2Pitch[i] = src2Pitch[i];
    eCondPitch[i] = condPitch[i];
  }

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              size_t src1I = x * eSrc1Pitch[0] + y * eSrc1Pitch[1] +
                             z * eSrc1Pitch[2] + w * eSrc1Pitch[3] +
                             q * eSrc1Pitch[4] + r * eSrc1Pitch[5];
              size_t src2I = x * eSrc2Pitch[0] + y * eSrc2Pitch[1] +
                             z * eSrc2Pitch[2] + w * eSrc2Pitch[3] +
                             q * eSrc2Pitch[4] + r * eSrc2Pitch[5];
              size_t dstI = x * eDstPitch[0] + y * eDstPitch[1] +
                            z * eDstPitch[2] + w * eDstPitch[3] +
                            q * eDstPitch[4] + r * eDstPitch[5];
              size_t condI = x * eCondPitch[0] + y * eCondPitch[1] +
                             z * eCondPitch[2] + w * eCondPitch[3] +
                             q * eCondPitch[4] + r * eCondPitch[5];
              ptrDstT[dstI] =
                  (ptrCondT[condI]) ? ptrSrcT1[src1I] : ptrSrcT2[src2I];
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
inline void fwdLibElementSelectInstThreaded(LibTensor* outT, LibTensor* condT,
                                            LibTensor* in1T, LibTensor* in2T,
                                            uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer<void>();
  void* srcT1 = in1T->getRawDataPointer<void>();
  void* srcT2 = in2T->getRawDataPointer<void>();
  
  // Addresser<srcType> ptrDstT(dstT, scale[3], offset[3]);
  Addresser<srcType> ptrDstT(dstT, outT->getScale(), outT->getOffset());
  // bool *ptrCondT = (bool *)condT;
  bool* ptrCondT = condT->getRawDataPointer<bool>();
  // const Addresser<srcType> ptrSrcT1(srcT1, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT1(srcT1, in1T->getScale(), in1T->getOffset());
  // const Addresser<srcType> ptrSrcT2(srcT2, scale[2], offset[2]);
  const Addresser<srcType> ptrSrcT2(srcT2, in2T->getScale(), in2T->getOffset());
  
  // unsigned int *actIndex = (unsigned int *)srcDims;
  const dim_t *actIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *act1Pitch = (unsigned int *)src1Pitches;
  const dim_t *act1Pitch = in1T->strides().data();
  // unsigned int *act2Pitch = (unsigned int *)src2Pitches;
  const dim_t *act2Pitch = in2T->strides().data();
  // unsigned int *condPitch = (unsigned int *)condPitches;
  const dim_t *condPitch = condT->strides().data();
  
  unsigned int srcDimNum = static_cast<unsigned int>(in1T->ndims());

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  unsigned int offsetIn1 = 0;
  unsigned int offsetIn2 = 0;
  unsigned int offsetOut = 0;
  unsigned int offsetCond = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn1 += act1Pitch[j] * coord[j];
    offsetIn2 += act2Pitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
    offsetCond += condPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    ptrDstT[offsetOut] =
        (ptrCondT[offsetCond]) ? ptrSrcT1[offsetIn1] : ptrSrcT2[offsetIn2];
    done = getOffsets(srcDimNum, coord, offsetIn1, offsetIn2, offsetOut,
       offsetCond, actIndex, act1Pitch, act2Pitch, dstPitch, condPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ELEMENT_SELECT_INST_H_
