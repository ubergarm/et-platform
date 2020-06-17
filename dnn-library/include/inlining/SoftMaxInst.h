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

#ifndef _SOFTMAX_INST_H_
#define _SOFTMAX_INST_H_

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
#include "SoftMaxInst1.h" // From include/inlining path
#include "SoftMaxInst2.h" // From include/inlining path
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
inline void fwdLibSoftMaxInst(LibTensor* outT, LibTensor* inT,
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

    // Find Max.
    float max = float(tInput[start]);
    for (unsigned int i = start + 1; i < end; i++)
      max = std::max(max, float(tInput[i]));

    // Compute exp.
    sum = 0;
    for (unsigned int i = 0; i < srcIndex[1]; i++) {
      e = getExp(float(tInput[n * srcPitch[0] + i]) - max);
      sum += e;
      tOutput[n * srcPitch[0] + i] = float(e);
    }

    fpReciprocalSingleElement(sum, inverseSum);
    // Normalize the output.
    for (unsigned int i = 0; i < srcIndex[1]; i++) {
      auto in = acumInt[n * srcPitch[0] + i];
      in = in * inverseSum;
      tOutput[n * srcPitch[0] + i] = in;
    }
  }
}

template <ElemKind elK>
inline void fwdLibSoftMaxInstThreaded(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  // unsigned int *srcPitch = (unsigned int *)srcTPitches;
  const dim_t *srcPitch = inT->strides().data();

  size_t typeSize = getsize<srcType>();
  unsigned int cll = CACHE_LINE_BYTES/typeSize;
  if (srcPitch[0]%cll == 0)
    dnn_lib::inlining::fwdLibSoftMaxInstThreaded1<elK>(outT, inT, flags, minionOffset, assignedMinions);
  else if (cll%srcPitch[0] == 0)
    dnn_lib::inlining::fwdLibSoftMaxInstThreaded2<elK>(outT, inT, flags, minionOffset, assignedMinions);
  else
    dnn_lib::inlining::fwdLibSoftMaxInst2<elK>(outT, inT, flags, minionOffset, assignedMinions);

}

template <ElemKind elK>
inline void fwdLibSoftMaxInstVectorized(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                        const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  // unsigned int *srcPitch = (unsigned int *)srcTPitches;
  const dim_t *srcPitch = inT->strides().data();
 
  size_t typeSize = getsize<srcType>();
  unsigned int cll = CACHE_LINE_BYTES/typeSize;
  if (srcPitch[0]%cll == 0)
    dnn_lib::inlining::fwdLibSoftMaxInstVectorized1<elK>(outT, inT, flags, minionOffset, assignedMinions);
  else if (cll%srcPitch[0] == 0) // TODO: vectorize v2.
    dnn_lib::inlining::fwdLibSoftMaxInstThreaded2<elK>(outT, inT, flags, minionOffset, assignedMinions);
  else
    dnn_lib::inlining::fwdLibSoftMaxInst2<elK>(outT, inT, minionOffset, assignedMinions); 
}

} // namespace inlining

} // namespace dnn_lib

#endif // _SOFTMAX_INST_H_
