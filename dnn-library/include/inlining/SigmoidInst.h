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

#ifndef _SIGMOID_INST_H_
#define _SIGMOID_INST_H_

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
inline void fwdLibSigmoidInstThreaded(LibTensor* outT, LibTensor* inT, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;
 
  /* maintain compatibility through the new Iface Libtensor */

  void* srcT1 = reinterpret_cast<void*>(inT->getUnsafePtr());
  void* dstT = reinterpret_cast<void*>(outT->getUnsafePtr());

  // Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);
  Addresser<srcType> ptrDstT(dstT, outT->dbggetscale(), outT->dbggetoffset());
  // const Addresser<srcType> ptrSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> ptrSrcT1(srcT1, inT->dbggetscale(), inT->dbggetoffset());

  // unsigned int *actIndex = (unsigned int *)srcDims;
  dim_t actIndex[max_tensor_dimensions] = {0,};
  outT->dims(actIndex);
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  dim_t dstPitch[max_tensor_dimensions] = {0,};
  outT->dbgcpypitches(dstPitch);
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  dim_t actPitch[max_tensor_dimensions] = {0,};
  inT->dbgcpypitches(actPitch);

  unsigned int srcDimNum = static_cast<unsigned int>(inT->dbggetnumdims());

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

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  float op, inverse;
  while (!done && (offsetOut < posMax)) {
    op = getExp(-ptrSrcT1[offsetIn]) + 1.0;
    fpReciprocalSingleElement(op, inverse);
    ptrDstT[offsetOut] = inverse;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename srcType>
inline void fwdLibSigmoidInst(LibTensor* outT, LibTensor* inT) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */

  void* srcT1 = reinterpret_cast<void*>(inT->getUnsafePtr());
  void* dstT = reinterpret_cast<void*>(outT->getUnsafePtr());
  
  // Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);
  Addresser<srcType> ptrDstT(dstT, outT->dbggetscale(), outT->dbggetoffset());  
  // const Addresser<srcType> ptrSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> ptrSrcT1(srcT1, inT->dbggetscale(), inT->dbggetoffset());
  
  // unsigned int *srcIndex = (unsigned int *)srcDims;
  dim_t srcIndex[max_tensor_dimensions] = {0,};
  inT->dims(srcIndex); 
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  dim_t dstPitch[max_tensor_dimensions] = {0,};
  outT->dbgcpypitches(dstPitch);
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  dim_t srcPitch[max_tensor_dimensions] =  {0,};
  inT->dbgcpypitches(srcPitch);
  
  unsigned int srcDimNum = static_cast<unsigned int>(inT->dbggetnumdims());
  
  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;
  float op, inverse;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              op = getExp(-ptrSrcT1[addrSrc]) + 1.0;
              fpReciprocalSingleElement(op, inverse);
              ptrDstT[addrDst] = inverse;
            }
          }
        }
      }
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _SIGMOID_INST_H_

