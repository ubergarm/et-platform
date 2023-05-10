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
INLINE_ATTR void fwdLibExtractTensorInst(LibTensor* outT, LibTensor* inT, const dim_array_t& coord, uint64_t flags,
                                         const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer();
  void* src = inT->getRawDataPointer();

  // Addresser<elK> tOutput(dst, scale[1], offset[1]);
  Addresser<elK> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tInput(src, scale[0], offset[0]);
  const Addresser<elK> tInput(src, inT->getScale(), inT->getOffset());
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = inT->strides().data();

  dim_t dstDimNum = outT->ndims();

  size_t numElemsDst = dstPitch[0] * dstIndex[0];
  size_t initialaddrOut, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialaddrOut, maxRead,
                        minionId, activeMinions, dst);
  if (maxRead == 0)
    return;

  dim_array_t coordOut = {0};
  dim_t k;
  getNonPaddingCoordinates(coordOut, initialaddrOut, dstDimNum, dstPitch,
                           dstIndex, k);

  size_t offsetOut = 0;
  for (size_t i = 0; i < k; i++)
    offsetOut += dstPitch[i] * coordOut[i];
  size_t offsetIn = 0;
  for (dim_t i = 0; i < dstDimNum; ++i)
    offsetIn += (coord[i] + coordOut[i]) * srcPitch[i];

  auto posMaxOut = maxRead + initialaddrOut;
  bool done = false;
  while (!done && (offsetOut < posMaxOut)) {
    tOutput[offsetOut] = tInput[offsetIn];
    done = getOffsets(dstDimNum, coordOut, offsetIn, offsetOut, dstIndex,
                      srcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialaddrOut, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _EXTRACT_TENSOR_INST_H_
