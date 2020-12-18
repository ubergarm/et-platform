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


template <ElemKind elK>
inline void fwdLibElementSelectInst(LibTensor* outT, LibTensor* condT,
                                    LibTensor* in1T, LibTensor* in2T,
                                    uint64_t flags,
                                    const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer<void>();
  void* srcT1 = in1T->getRawDataPointer<void>();
  void* srcT2 = in2T->getRawDataPointer<void>();
  
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
  
  unsigned int srcDimNum = static_cast<unsigned int>(in1T->ndims());

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
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
  unsigned int clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ELEMENT_SELECT_INST_H_
