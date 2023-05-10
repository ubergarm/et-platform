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

#ifndef _QUANTIZED_INST_H_
#define _QUANTIZED_INST_H_

#include "Addresser.h"
#include "Float16.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

namespace dnn_lib {

namespace inlining {

template <ElemKind dstElK, ElemKind srcElK>
INLINE_ATTR void fwdLibQuantizeInst(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset = 0,
                                    const uint32_t assignedMinions = 0) {
  using dstType = typename elemKind2elemTy<dstElK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer();

  Addresser<dstElK> ptrDstT(dstT, outT->getScale(), outT->getOffset());
  const Addresser<srcElK> ptrSrcT(inT->getRawDataPointer(), inT->getScale(), inT->getOffset());

  const dim_t *srcIndex = inT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *srcPitch = inT->strides().data();
  dim_t srcDimNum = inT->ndims();

  size_t numElemsDst = dstPitch[0] * srcIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  size_t initialAddr, maxRead;
  size_t typeSize = sizeof(dstType);
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position
  dim_array_t coord = {0}; // Vector of coordinates
  dim_t k = 0;             // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, srcIndex, k);

  // We get the actual initialAddr, in the input and output.
  size_t offsetIn = 0;
  size_t offsetOut = 0;
  for (dim_t j = 0; j < k; j++) {
    offsetIn += srcPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  size_t posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done) {
    if (offsetOut >= posMax) break;
    auto val = ptrSrcT[offsetIn];
    ptrDstT[offsetOut] = val;

    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, srcIndex,
                      srcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _QUANTIZED_INST_H_
