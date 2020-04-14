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

template <typename srcType>
inline void fwdLibModuloInst(LibTensor* inT, LibTensor* outT, long long divisor,
                               bool signFollowDivisor) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */

  auto srcH = inT->getHandle<srcType>();
  auto destH = outT->getHandle<srcType>();

  //  srcType *tOutput = (srcType *)dstT;
  srcType *tOutput = (srcType *)destH.getUnsafePtrdbg();
  //  srcType *tInput = (srcType *)srcT;  
  srcType *tInput = (srcType *)srcH.getUnsafePtrdbg();

  //  unsigned int *dstIndex = (unsigned int *)dstDims;
  dim_t dstIndex[max_tensor_dimensions] = {0,};
  destH.cpydims(dstIndex);
  // unsigned int *srcIndex = (unsigned int *)srcDims;
  dim_t srcIndex[max_tensor_dimensions] = {0,};
  srcH.cpydims(srcIndex);

  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  dim_t dstPitch[max_tensor_dimensions] = {0,};
  destH.cpypitchesdbg(dstPitch);
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  dim_t srcPitch[max_tensor_dimensions] = {0,};
  srcH.cpypitchesdbg(srcPitch);
  
  // unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  // unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  // unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  dim_t eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  dim_t eDstPitch[MAX_TENSOR_DIMENSIONS] = {0,};
  dim_t eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0,};
  
  unsigned int srcDimNum = static_cast<unsigned int>(srcH.getNumDimsdbg());
  
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

template <typename srcType>
inline void fwdLibModuloInstThreaded(LibTensor* inT, LibTensor* outT,long long divisor,
                                     bool signFollowDivisor, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  auto srcH = inT->getHandle<srcType>();
  auto dstH = outT->getHandle<srcType>();

  void* srcT = reinterpret_cast<void*>(srcH.getUnsafePtrdbg());
  void* dstT = reinterpret_cast<void*>(dstH.getUnsafePtrdbg());
   
  // Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  // const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, outT->dbggetscale(), outT->dbggetoffset());
  const Addresser<srcType> tInput(srcT, inT->dbggetscale(), inT->dbggetoffset());
 
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  dim_t dstIndex[max_tensor_dimensions] = {0,};
  dstH.cpydims(dstIndex);
  // unsigned int *actIndex = (unsigned int *)srcDims;
  dim_t actIndex[max_tensor_dimensions] = {0,};
  srcH.cpydims(actIndex);

  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  dim_t dstPitch[max_tensor_dimensions] = {0,};
  dstH.cpypitchesdbg(dstPitch);

  // unsigned int *actPitch = (unsigned int *)srcPitches;
  dim_t actPitch[max_tensor_dimensions] = {0,};
  srcH.cpypitchesdbg(actPitch);

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int srcDimNum = static_cast<unsigned int>(inT->dbggetnumdims());
  
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
