/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _MODULO_INST_H_
#define _MODULO_INST_H_

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
INLINE_ATTR void fwdLibModuloInst(LibTensor* outT, LibTensor* inT, uint64_t divisor, bool signFollowDivisor,
                                  uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */

  void* srcT = inT->getRawDataPointer();
  void* dstT = outT->getRawDataPointer();

  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  const Addresser<elK> tInput(srcT, inT->getScale(), inT->getOffset());

  const dim_t *actIndex = inT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *actPitch = inT->strides().data();

  size_t numElemsDst = dstPitch[0] * actIndex[0];
  size_t initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  dim_t srcDimNum = inT->ndims();

  dim_array_t coord = {0};
  dim_t k;

  /* overloading while sw-2400 and sw-2429 are WIP */
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex, k);

  size_t offsetIn = 0;
  size_t offsetOut = 0;
  for (dim_t j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  size_t posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    auto res = (tInput[offsetIn]) % static_cast<srcType>(divisor);
    if (signFollowDivisor && (res < 0)) {
      res += static_cast<srcType>(divisor);
    }
    tOutput[offsetOut] = res;
    
    /* overloading while sw-2400 and sw-2429 are WIP */    
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex, actPitch,
                      dstPitch);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namspace dnn_lib

#endif // _MODULO_INST_H_
