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

#ifndef _EXTRACT_TENSOR_INST_H_
#define _EXTRACT_TENSOR_INST_H_

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
inline void fwdLibExtractTensorInst(LibTensor* outT, LibTensor* inT,
                                    void *pcoord,
                                    uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  if (get_minion_id() != minionOffset) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* src = inT->getRawDataPointer<void>();
  
  // Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  Addresser<srcType> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tInput(src, scale[0], offset[0]);
  const Addresser<srcType> tInput(src, inT->getScale(), inT->getOffset());

  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = inT->strides().data();
  
  unsigned int *coord = (unsigned int *)pcoord;

  unsigned int dstDimNum = static_cast<unsigned int>(outT->ndims());
  
  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eOffsets[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < dstDimNum; i++) {
    eDims[i] = dstIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
    eOffsets[i] = coord[i];
  }

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              tOutput[x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                      w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5]] =
                  tInput[(eOffsets[0] + x) * eSrcPitch[0] +
                         (eOffsets[1] + y) * eSrcPitch[1] +
                         (eOffsets[2] + z) * eSrcPitch[2] +
                         (eOffsets[3] + w) * eSrcPitch[3] +
                         (eOffsets[4] + q) * eSrcPitch[4] +
                         (eOffsets[5] + r) * eSrcPitch[5]];
            }
          }
        }
      }
    }
  }
}

template <ElemKind elK>
inline void fwdLibExtractTensorInstThreaded(LibTensor* outT, LibTensor* inT,
                                            void *pcoord, uint64_t flags,
                                            const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* src = inT->getRawDataPointer<void>();

  // Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  Addresser<srcType> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tInput(src, scale[0], offset[0]);
  const Addresser<srcType> tInput(src, inT->getScale(), inT->getOffset());
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = inT->strides().data();
  
  unsigned int *coord = (unsigned int *)pcoord;

  unsigned int dstDimNum = static_cast<unsigned int>(outT->ndims());
  
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialaddrOut, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialaddrOut, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coordOut[dstDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coordOut, initialaddrOut, dstDimNum, dstPitch,
                           dstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++)
    offsetOut += dstPitch[i] * coordOut[i];
  unsigned int offsetIn = 0;
  for (unsigned int i = 0; i < dstDimNum; ++i)
    offsetIn += (coord[i] + coordOut[i]) * srcPitch[i];

  unsigned int posMaxOut = maxRead + initialaddrOut;
  bool done = false;
  while (!done && (offsetOut < posMaxOut)) {
    tOutput[offsetOut] = tInput[offsetIn];
    done = getOffsets(dstDimNum, coordOut, offsetIn, offsetOut, dstIndex,
                      srcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialaddrOut, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _EXTRACT_TENSOR_INST_H_
