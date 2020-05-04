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

#ifndef _EMBEDDING_BAG_INST_H_
#define _EMBEDDING_BAG_INST_H_

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

// Notes :
//
// dim_t in glow is expected to be uint64_t
// sdim_t in glow is expected to be sint64_t
//
//
// dst tensor first dimension should be >= dimension of offset tensor
// dst tensor second+ dimensions should match data tensor second+ dimensions
// 
// As there is no checks in the implementation at the moment only data,
// indices and offsets tensors dimensions are required.
//

inline void fwdLibEmbeddingBagInstFloatTy(LibTensor* outT, LibTensor *in1T,
                                          LibTensor* in2T, LibTensor* in3T,
                                          LibTensor* in4T, uint64_t dataDim1Pitch) {

  uint32_t minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */
  // float *dst     = (float *) pdst;
  float *dst = outT->getRawDataPointer<float>();
  // float *data    = (float *) pdata;
  float *data = in1T->getRawDataPointer<float>();
  // float *weights = (float *) pweights;
  float *weights = in2T->getRawDataPointer<float>();
  // uint64_t *indices = (uint64_t *) pindices;
  uint64_t *indices = in3T->getRawDataPointer<uint64_t>();
  // uint64_t *offsets = (uint64_t *) poffsets;
  uint64_t *offsets = in4T->getRawDataPointer<uint64_t>();

  // const size_t segments    = offsetsSize;
    const size_t segments = in4T->dims().data()[0];
  // const size_t totalLength = indicesSize;
  const size_t totalLength = in3T->dims().data()[0];

  // NOTE : Pitch is passed as the number of elements, not as bytes.
  const size_t lineSize = dataDim1Pitch;
  //@TODO in SW-2429 remove dataDim1Pitch param once the instruction bellow It works. 
  //const size_t lineSize = (in1T->strides().data()[0]/in1T->getElementSize());
  
  uint64_t curIdx = 0;
  for (uint64_t i = 0; i < segments; i++) {
    uint64_t start = offsets[i];
    uint64_t end = (i == (segments - 1)) ? totalLength : offsets[i + 1];
    // Unroll the first iteration as the dst tensor needs to be
    // initialized.

    // The offsets can be defined so that a segment is empty with
    // the effect of forcing to zero the output segment.
    const float  weight    = (start < end) ? weights[curIdx] : 0;
    size_t       offsetIn  = indices[curIdx] * lineSize;
    size_t       offsetOut = i * lineSize;
    if (start < end)
      curIdx++;

    for (uint64_t k = 0; k < lineSize; k++)
      dst[offsetOut++] = data[offsetIn++] * weight;

    for (uint64_t j = (start + 1); j < end; j++) {
      const float weight    = weights[curIdx];
      size_t      offsetIn  = indices[curIdx++] * lineSize;
      size_t      offsetOut = i * lineSize;
      for (size_t k = 0; k < lineSize; k++) 
        dst[offsetOut++] += data[offsetIn++] * weight;
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _EMBEDDING_BAG_INST_H_
