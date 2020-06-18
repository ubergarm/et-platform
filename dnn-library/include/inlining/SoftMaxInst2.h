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

#ifndef _SOFTMAX_INST2_H_
#define _SOFTMAX_INST2_H_

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

// Single-thread version with small optimisations. Useful when the padding
// hypothesis are not met.
template <ElemKind elK>
inline void fwdLibSoftMaxInst2(LibTensor* outT, LibTensor* inT,
                               uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  if (get_minion_id() != minionOffset) return;
  /* maintain compatibility through the new Iface Libtensor */  
  srcType* dstT = outT->getRawDataPointer<srcType>();
  srcType* srcT = inT->getRawDataPointer<srcType>();

  // Addresser<elK> tOutput(dstT, scale[1], offset[1]);
  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  // const Addresser<elK> acumInt(dstT, scale[1], offset[1]);
  const Addresser<elK> acumInt(dstT, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tInput(srcT, scale[0], offset[0]);
  const Addresser<elK> tInput(srcT, inT->getScale(), inT->getOffset());
  
  // unsigned int *srcIndex = (unsigned int *)srcTDims;
  const dim_t *srcIndex = inT->dims().data();
  // unsigned int *srcPitch = (unsigned int *)srcTPitches;
  const dim_t *srcPitch = inT->strides().data();
  
  float e, sum, inverseSum;

  for (unsigned int n = 0; n < srcIndex[0]; n++) {
    unsigned int start = n * srcPitch[0];
    unsigned int end = start + srcIndex[1];

    float max = float(tInput[start]);
    for (unsigned int i = start + 1; i < end; i++)
      max = std::max(max, float(tInput[i]));

    // Compute exp.
    sum = 0;
    for (unsigned int i = start; i < end; i++) {
      e = getExp(float(tInput[i]) - max);
      sum += e;
      tOutput[i] = float(e); // here, the shape hypothesis is important.
    }

    fpReciprocalSingleElement(sum, inverseSum);
    // Normalize the output.
    for (unsigned int i = start; i < end; i++) {
      auto in = acumInt[i];
      in = in * inverseSum;
      tOutput[i] = in;
    }
  }
}

template <ElemKind elK>
inline void fwdLibSoftMaxInstThreaded2(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                       const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  srcType* dstT = outT->getRawDataPointer<srcType>();
  srcType* srcT = inT->getRawDataPointer<srcType>();
   
  // Addresser<elK> tOutput(dstT, scale[1], offset[1]);
  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  // const Addresser<elK> acumInt(dstT, scale[1], offset[1]);
  const Addresser<elK> acumInt(dstT, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tInput(srcT, scale[0], offset[0]);
  const Addresser<elK> tInput(srcT, inT->getScale(), inT->getOffset());
 
  // unsigned int *srcIndex = (unsigned int *)srcTDims;
  const dim_t *srcIndex = inT->dims().data();
  // unsigned int *srcPitch = (unsigned int *)srcTPitches;
  const dim_t *srcPitch = inT->strides().data();
  
  size_t typeSize = getsize<srcType>();
  unsigned int cll = CACHE_LINE_BYTES/typeSize;
  unsigned int rowspercl = (cll - 1)/srcPitch[0] + 1;
  unsigned int rowstodo = rowspercl;
  while(activeMinions*rowstodo < srcIndex[0]) rowstodo += rowspercl;

  unsigned int firstrow = minionId*rowstodo;
  if (firstrow >= srcIndex[0])
    return;
  unsigned int lastrow = firstrow + rowstodo;
  if (lastrow > srcIndex[0])
    lastrow = srcIndex[0];

  float e, sum, inverseSum;

  for (unsigned int n = firstrow; n < lastrow; n++) {
    unsigned int start = n * srcPitch[0];
    unsigned int end = start + srcIndex[1];

    float max = float(tInput[start]);
    for (unsigned int i = start + 1; i < end; i++)
      max = std::max(max, float(tInput[i]));

    // Compute exp.
    sum = 0;
    for (unsigned int i = start; i < end; i++) {
      e = getExp(float(tInput[i]) - max);
      sum += e;
      tOutput[i] = float(e); // here, the shape hypothesis is important.
    }

    fpReciprocalSingleElement(sum, inverseSum);
    // Normalize the output.
    for (unsigned int i = start; i < end; i++) {
      auto in = acumInt[i];
      in = in * inverseSum;
      tOutput[i] = in;
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _SOFTMAX_INST2_H_
