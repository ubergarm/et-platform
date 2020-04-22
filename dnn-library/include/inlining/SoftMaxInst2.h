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
template <typename srcType>
inline void fwdLibSoftMaxInst2(LibTensor* outT, LibTensor* inT) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;
  
  auto dstH = outT->getHandle<srcType>();
  auto srcH = inT->getHandle<srcType>();
  
  srcType* dstT = reinterpret_cast<srcType*>(dstH.getUnsafePtrdbg());
  srcType* srcT = reinterpret_cast<srcType*>(srcH.getUnsafePtrdbg());
 
  // Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  Addresser<srcType> tOutput(dstT, dstH.getScaledbg(), dstH.getOffsetdbg());
  // const Addresser<srcType> acumInt(dstT, scale[1], offset[1]);
  const Addresser<srcType> acumInt(dstT, dstH.getScaledbg(), dstH.getOffsetdbg());
  // const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  const Addresser<srcType> tInput(srcT, srcH.getScaledbg(), srcH.getOffsetdbg());
  
  // unsigned int *srcIndex = (unsigned int *)srcTDims;
  dim_t srcIndex[max_tensor_dimensions] = {0,};
  dstH.cpydims(srcIndex);
  // unsigned int *srcPitch = (unsigned int *)srcTPitches;
  dim_t srcPitch[max_tensor_dimensions] = {0,};
  srcH.cpypitchesdbg(srcPitch);
 
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

template <typename srcType>
inline void fwdLibSoftMaxInstThreaded2(LibTensor* outT, LibTensor* inT, uint64_t flags) {
  
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

 auto dstH = outT->getHandle<srcType>();
  auto srcH = inT->getHandle<srcType>();
  
  srcType* dstT = reinterpret_cast<srcType*>(dstH.getUnsafePtrdbg());
  srcType* srcT = reinterpret_cast<srcType*>(srcH.getUnsafePtrdbg());
  
  // Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  Addresser<srcType> tOutput(dstT, dstH.getScaledbg(), dstH.getOffsetdbg());
  // const Addresser<srcType> acumInt(dstT, scale[1], offset[1]);
  const Addresser<srcType> acumInt(dstT, dstH.getScaledbg(), dstH.getOffsetdbg());
  // const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  const Addresser<srcType> tInput(srcT, srcH.getScaledbg(), srcH.getOffsetdbg());
 
  // unsigned int *srcIndex = (unsigned int *)srcTDims;
  dim_t srcIndex[max_tensor_dimensions] = {0,};
  dstH.cpydims(srcIndex);
  // unsigned int *srcPitch = (unsigned int *)srcTPitches;
  dim_t srcPitch[max_tensor_dimensions] = {0,};
  srcH.cpypitchesdbg(srcPitch);
  
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
