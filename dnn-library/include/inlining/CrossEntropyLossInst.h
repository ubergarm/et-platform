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
inline __attribute__((always_inline)) void fwdLibCrossEntropyLossInst(LibTensor* outT,
                                                                      LibTensor* in1T,
                                                                      LibTensor* in2T) {
  using srcType = typename elemKind2elemTy<elK>::type;
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* outT --> dst  in1T--> src in2T--> index*/
  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* src = in1T->getRawDataPointer<void>();

  // Addresser<srcType> tOutput(dstT, scale[2], offset[2]);
  Addresser<srcType> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tTmp(dstT, scale[2], offset[2]);
  const Addresser<srcType> tTmp(dst, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  const Addresser<srcType> tInput(src, in1T->getScale(), in1T->getOffset());
  // long long *tLabels = (long long *)labelsT;
  long long *tLabels = in2T->getRawDataPointer<long long>();
  
  // unsigned int *srcIndex = (unsigned int *)srcDims;
  const dim_t *srcIndex = in1T->dims().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = in1T->strides().data();
  
  float op1;
  const float op2 = M_1_LOG2E;

  // Initialize to zero output tensor
  tOutput[0] = 0;

  for (size_t n = 0; n < srcIndex[0]; ++n) {
    size_t y = tLabels[n];
    float p_n = float(tInput[n * srcPitch[0] + y]);
    fpLog2SingleElement(p_n, op1);
    float mulOp = op1 * op2;
    auto tmp = tTmp[0];
    tmp -= mulOp;
    tOutput[0] = tmp;
  }
}

template <ElemKind elK>
inline __attribute__((always_inline)) void fwdLibCrossEntropyLossInstThreaded(
                                                                              LibTensor* outT,
                                                                              LibTensor* in1T,
                                                                              LibTensor* in2T,
                                                                              uint64_t flags) {
  using srcType = typename elemKind2elemTy<elK>::type;
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  void* src = in1T->getRawDataPointer<void>();

  // Addresser<srcType> tOutput(dstT, scale[2], offset[2]);
  Addresser<srcType> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tTmp(dstT, scale[2], offset[2]);
  const Addresser<srcType> tTmp(dst, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  const Addresser<srcType> tInput(src, in1T->getScale(), in1T->getOffset());
  // long long *tLabels = (long long *)labelsT;
  long long *tLabels = in2T->getRawDataPointer<long long>();
  
  // unsigned int *srcIndex = (unsigned int *)srcDims;
  const dim_t *srcIndex = in1T->dims().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = in1T->strides().data();
 
  unsigned int rowstodo = srcIndex[0] / activeMinions;
  unsigned int firstrow;
  unsigned int type1minions = srcIndex[0] - rowstodo * activeMinions;
  if (minionId < type1minions) {
    ++rowstodo;
    firstrow = minionId * rowstodo;
  } else
    firstrow = type1minions +
               minionId * rowstodo; // Simplification of type1minions*(rowstodo
                                    // + 1) + (minionId - type1minions)*rowstodo
  unsigned int lastrow = firstrow + rowstodo;

  float op;
  float sum = 0;
  unsigned int rowaddress =
      firstrow * srcPitch[0]; // address of the first element of the row
                              // considered in the following loop.
  for (size_t n = firstrow; n < lastrow; ++n) {
    float p_n = float(tInput[rowaddress + tLabels[n]]);
    fpLog2SingleElement(p_n, op);
    sum -= op;
    rowaddress += srcPitch[0];
  }

  unsigned int level = 0;
  for (unsigned int k = 1; k < activeMinions; k *= 2)
    level++;

  for (unsigned int i = 0; i < level; i++)
    sum = tensor_reduce_float(sum, 0x0, 1, i, 0x3);
  if (minionId == 0)
    tOutput[0] = sum * M_1_LOG2E;
}

} // namespace inlining

} // namespace dnn_lib

#endif // _CROSS_ENTROPY_LOST_INST_H_
