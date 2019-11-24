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

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "LibNodes.h"
#include "GenInstances.h"
#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"

using namespace std;

template <typename srcType>
void dnn_lib::fwdLibSoftMaxInst(void *dstT, void *srcT, void *srcTDims,
                                void *srcTPitches, float *scale,
                                int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  const Addresser<srcType> acumInt(dstT, scale[1], offset[1]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);

  unsigned int *srcIndex = (unsigned int *)srcTDims;
  unsigned int *srcPitch = (unsigned int *)srcTPitches;

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

template <typename srcType>
void dnn_lib::fwdLibSoftMaxInstThreaded(void *dstT, void *srcT, void *srcTDims,
                                        void *srcTPitches, float *scale,
                                        int32_t *offset, uint64_t flags) {

  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  size_t typeSize = getsize<srcType>();
  unsigned int cll = 64/typeSize;
  if (srcPitch[0]%cll == 0)
    fwdLibSoftMaxInstThreaded1<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset, flags);
  else if (cll%srcPitch[0] == 0)
    fwdLibSoftMaxInstThreaded2<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset, flags);
  else fwdLibSoftMaxInst2<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset);
}

template <typename srcType>
void dnn_lib::fwdLibSoftMaxInstVectorized(void *dstT, void *srcT, void *srcTDims,
                                          void *srcTPitches, float *scale,
                                          int32_t *offset, uint64_t flags) {

  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  size_t typeSize = getsize<srcType>();
  unsigned int cll = 64/typeSize;
  if (srcPitch[0]%cll == 0)
    fwdLibSoftMaxInstVectorized1<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset, flags);
  else if (cll%srcPitch[0] == 0) // TODO: vectorize v2.
    fwdLibSoftMaxInstThreaded2<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset, flags);
  else fwdLibSoftMaxInst2<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset);
}

GEN_INSTANCES_OP(template, fwdLibSoftMaxInst, void *dstT, void *srcT, void *srcTDims,
                          void *srcTPitches, float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibSoftMaxInstThreaded, void *dstT, void *srcT, void *srcTDims,
                          void *srcTPitches, float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibSoftMaxInstVectorized, void *dstT, void *srcT, void *srcTDims,
                          void *srcTPitches, float *scale, int32_t *offset, uint64_t flags);
