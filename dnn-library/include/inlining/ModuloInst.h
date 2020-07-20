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

#ifndef _MODULO_INST_H_
#define _MODULO_INST_H_

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
inline void fwdLibModuloInst(LibTensor* outT, LibTensor* inT, long long divisor,
                             bool signFollowDivisor,
                             uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  
  if (get_minion_id() != minionOffset) return;

  /* maintain compatibility through the new Iface Libtensor */
 
  //  srcType *tOutput = (srcType *)dstT;
  srcType *tOutput = outT->getRawDataPointer<srcType>();
  //  srcType *tInput = (srcType *)srcT;  
  srcType *tInput = inT->getRawDataPointer<srcType>();

  //  unsigned int *dstIndex = (unsigned int *)dstDims;

  // unsigned int *srcIndex = (unsigned int *)srcDims;
  const dim_t *srcIndex = inT->dims().data();
  
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = inT->strides().data();
  
  // unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  // unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  // unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  dim_t eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  dim_t eDstPitch[MAX_TENSOR_DIMENSIONS] = {0,};
  dim_t eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0,};
  
  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  for (size_t i = 0; i < srcDimNum; i++) {
    eDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;

  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              auto res = (tInput[addrSrc]) % divisor;
              if (signFollowDivisor && (res < 0)) {
                res += divisor;
              }
              tOutput[addrDst] = res;
            }
          }
        }
      }
    }
  }
}

template <ElemKind elK>
inline void fwdLibModuloInstThreaded(LibTensor* outT, LibTensor* inT,long long divisor,
                                     bool signFollowDivisor, uint64_t flags,
                                     const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */

  void* srcT = inT->getRawDataPointer<void>();
  void* dstT = outT->getRawDataPointer<void>();
   
  // Addresser<elK> tOutput(dstT, scale[1], offset[1]);
  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tInput(srcT, scale[0], offset[0]);  
  const Addresser<elK> tInput(srcT, inT->getScale(), inT->getOffset());
 
  // unsigned int *dstIndex = (unsigned int *)dstDims;

  // unsigned int *actIndex = (unsigned int *)srcDims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  const dim_t *actPitch = inT->strides().data();
  
  unsigned int numElemsDst = dstPitch[0] * actIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  unsigned int coord[srcDimNum];
  unsigned int k;
  
  /* overloading while sw-2400 and sw-2429 are WIP */
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex, k);  
  
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    auto res = (tInput[offsetIn]) % divisor;
    if (signFollowDivisor && (res < 0)) {
      res += divisor;
    }
    tOutput[offsetOut] = res;
    
    /* overloading while sw-2400 and sw-2429 are WIP */    
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex, actPitch,
                      dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namspace dnn_lib

#endif // _MODULO_INST_H_
