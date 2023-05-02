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

#ifndef _INT_LOOKUP_TABLE_INST_H_
#define _INT_LOOKUP_TABLE_INST_H_

#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

#include "Float16.h"
#include "LibTensor.h"
#include "utils.h"

namespace dnn_lib {

namespace inlining {
INLINE_ATTR void fwdLibIntLookupTableInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, uint64_t flags,
                                          const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  assert(outT->getElementType() == Int8QTy &&
         in1T->getElementType() == Int8QTy &&
         in2T->getElementType() == Int8QTy);

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer();

  auto ptrDstT = outT->getRawDataPointer<int8_t>();
  auto ptrSrcT1 = in1T->getRawDataPointer<int8_t>();
  auto ptrSrcT2 = in2T->getRawDataPointer<int8_t>();

  const dim_t* dstIndex =  outT->dims().data();
  const dim_t* src1Index =  in1T->dims().data();
  const dim_t* dstPitch = outT->strides().data();
  const dim_t* src1Pitch = in1T->strides().data();

  dim_t dstDimNum = outT->ndims();

  size_t numElemsDst = dstPitch[0] * dstIndex[0];

  size_t initialAddr, maxRead;
  getCachelinePartition(sizeof(int8_t), numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, ptrDstT);
  if (maxRead == 0)
    return;

  dim_array_t coord = {0};
  dim_t k;
  /* overloading while sw-2400 and sw-2429 are WIP */
  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex, k);

  size_t offsetIn = 0;
  size_t offsetOut = 0;
  for (dim_t j = 0; j < k; j++) {
    offsetIn += src1Pitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  size_t posMax = maxRead + initialAddr;

  bool done = false;
  while (!done && (offsetOut < posMax)) {
    ptrDstT[offsetOut] = ptrSrcT2[ptrSrcT1[offsetIn] + 128];
    done = getOffsets(dstDimNum, coord, offsetIn, offsetOut, src1Index,
                      src1Pitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * sizeof(int8_t) + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + sizeof(int8_t)*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _INT_LOOKUP_TABLE_INST_H_
