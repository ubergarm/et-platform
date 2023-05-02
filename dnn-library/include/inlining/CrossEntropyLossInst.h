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

#ifndef _CROSS_ENTROPY_LOST_INST_H_
#define _CROSS_ENTROPY_LOST_INST_H_

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
inline __attribute__((always_inline)) void fwdLibCrossEntropyLossInst(
                                                                      LibTensor* outT,
                                                                      LibTensor* in1T,
                                                                      LibTensor* in2T,
                                                                      uint64_t flags,
                                                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer();
  void* src = in1T->getRawDataPointer();

  // Addresser<elK> tOutput(dstT, scale[2], offset[2]);
  Addresser<elK> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tTmp(dstT, scale[2], offset[2]);
  const Addresser<elK> tTmp(dst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tInput(srcT, scale[0], offset[0]);
  const Addresser<elK> tInput(src, in1T->getScale(), in1T->getOffset());
  // long long *tLabels = (long long *)labelsT;
  long long *tLabels = in2T->getRawDataPointer<long long>();
  
  // unsigned int *srcIndex = (unsigned int *)srcDims;
  const dim_t *srcIndex = in1T->dims().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = in1T->strides().data();

  auto rowstodo = srcIndex[0] / activeMinions;
  size_t firstrow;
  auto type1minions = srcIndex[0] - rowstodo * activeMinions;
  if (minionId < type1minions) {
    ++rowstodo;
    firstrow = minionId * rowstodo;
  } else
    firstrow = type1minions +
               minionId * rowstodo; // Simplification of type1minions*(rowstodo
                                    // + 1) + (minionId - type1minions)*rowstodo
  auto lastrow = firstrow + rowstodo;

  float op;
  float sum = 0;
  auto rowaddress = firstrow * srcPitch[0]; // address of the first element of the row
                                            // considered in the following loop.
  for (size_t n = firstrow; n < lastrow; ++n) {
    float p_n = float(tInput[rowaddress + tLabels[n]]);
    fpLog2SingleElement(p_n, op);
    sum -= op;
    rowaddress += srcPitch[0];
  }

  size_t level = 0;
  for (size_t k = 1; k < activeMinions; k *= 2) {
    level++;
  }

  for (size_t i = 0; i < level; i++) {
    sum = tensor_reduce_float(sum, 0x0, 1, i, 0x3);
  }

  if (minionId == 0) {
    static_assert((elK == FloatTy || elK == Float16Ty || elK == BFloat16Ty), "Unsupported elK type.");
    tOutput[0] = (sum * M_1_LOG2E);
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _CROSS_ENTROPY_LOST_INST_H_
