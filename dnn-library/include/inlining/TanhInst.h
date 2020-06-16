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

#ifndef _TANH_INST_H_
#define _TANH_INST_H_

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

// TODO Check corner cases
template <ElemKind elK>
inline void fwdLibTanhInst(LibTensor* outT, LibTensor* inT,
                           uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  
  if (get_minion_id() != minionOffset) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  void* src = inT->getRawDataPointer<void>();
  void* dst = outT->getRawDataPointer<void>();

  // const Addresser<srcType> ptrSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> ptrSrcT1(src, inT->getScale(), inT->getOffset());
  // Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);
  Addresser<srcType> ptrDstT(dst, outT->getScale(), outT->getOffset());
  
  // unsigned int *srcIndex = (unsigned int *)srcDims;
  const dim_t *srcIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t * dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = inT->strides().data();
  
  uint8_t srcDimNum = static_cast<uint8_t>(inT->ndims());
  
  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  float op1, op2;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              op1 = getSinh(ptrSrcT1[x * eSrcPitch[0] + y * eSrcPitch[1] +
                                     z * eSrcPitch[2] + w * eSrcPitch[3] +
                                     q * eSrcPitch[4] + r * eSrcPitch[5]]);
              op2 = getCosh(ptrSrcT1[x * eSrcPitch[0] + y * eSrcPitch[1] +
                                     z * eSrcPitch[2] + w * eSrcPitch[3] +
                                     q * eSrcPitch[4] + r * eSrcPitch[5]]);
              fpReciprocalSingleElement(op2, op2);
              ptrDstT[x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                      w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5]] =
                  op1 * op2;
            }
          }
        }
      }
    }
  }
}

template <ElemKind elK>
inline void fwdLibTanhInstThreaded(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                   const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  using srcType = typename elemKind2elemTy<elK>::type;

  /* maintain compatibility through the new Iface Libtensor */
  void* src = inT->getRawDataPointer<void>();
  void* dst = outT->getRawDataPointer<void>();

  // const Addresser<srcType> aSrcT1(src, scale[0], offset[0]);
  const Addresser<srcType> aSrcT1(src, inT->getScale(), inT->getOffset());
  // Addresser<srcType> ptrDstT(dst, scale[1], offset[1]);
  Addresser<srcType> ptrDstT(dst, outT->getScale(), outT->getOffset());
 
  // unsigned int *actIndex = (unsigned int *)srcDims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t * dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  const dim_t *actPitch = inT->strides().data();

  uint8_t srcDimNum = static_cast<uint8_t>(inT->ndims());

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;

  /* overloading while sw-2400 and sw-2429 are WIP */  
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  float op1, op2;
  while (!done && (offsetOut < posMax)) {
    op1 = getSinh(aSrcT1[offsetIn]);
    op2 = getCosh(aSrcT1[offsetIn]);
    fpReciprocalSingleElement(op2, op2);
    ptrDstT[offsetOut] = op1 * op2;

    /* overloading while sw-2400 and sw-2429 are WIP */    
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _TANH_INST_H_
