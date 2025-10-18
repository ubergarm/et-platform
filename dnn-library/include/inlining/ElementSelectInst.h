/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _ELEMENT_SELECT_INST_H_
#define _ELEMENT_SELECT_INST_H_

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
INLINE_ATTR void fwdLibElementSelectInst(LibTensor* outT, LibTensor* condT, LibTensor* in1T, LibTensor* in2T,
                                         [[maybe_unused]] uint64_t flags, const uint32_t minionOffset = 0,
                                         [[maybe_unused]] const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer();
  void* srcT1 = in1T->getRawDataPointer();
  void* srcT2 = in2T->getRawDataPointer();

  // Addresser<elK> ptrDstT(dstT, scale[3], offset[3]);
  Addresser<elK> ptrDstT(dstT, outT->getScale(), outT->getOffset());
  // bool *ptrCondT = (bool *)condT;
  bool* ptrCondT = condT->getRawDataPointer<bool>();
  // const Addresser<elK> ptrSrcT1(srcT1, scale[1], offset[1]);
  const Addresser<elK> ptrSrcT1(srcT1, in1T->getScale(), in1T->getOffset());
  // const Addresser<elK> ptrSrcT2(srcT2, scale[2], offset[2]);
  const Addresser<elK> ptrSrcT2(srcT2, in2T->getScale(), in2T->getOffset());
  
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

  dim_t srcDimNum = in1T->ndims();

  auto numElemsDst = dstPitch[0] * actIndex[0];

  size_t initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  dim_array_t coord = {0};
  dim_t k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  size_t offsetIn1 = 0;
  size_t offsetIn2 = 0;
  size_t offsetOut = 0;
  size_t offsetCond = 0;
  for (size_t j = 0; j < k; j++) {
    offsetIn1 += act1Pitch[j] * coord[j];
    offsetIn2 += act2Pitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
    offsetCond += condPitch[j] * coord[j];
  }

  auto posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    ptrDstT[offsetOut] =
        (ptrCondT[offsetCond]) ? ptrSrcT1[offsetIn1] : ptrSrcT2[offsetIn2];
    done = getOffsets(srcDimNum, coord, offsetIn1, offsetIn2, offsetOut,
       offsetCond, actIndex, act1Pitch, act2Pitch, dstPitch, condPitch);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ELEMENT_SELECT_INST_H_
