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

void dnn_lib::fwdLibEmbeddingBagInstFloatTy(void *pdst,
  void *pdata, uint64_t dataDim1Pitch,
  void *pweights,
  void *pindices, uint64_t indicesSize,
  void *poffsets, uint64_t offsetsSize) {

  uint32_t minionId = get_minion_id();
  if (minionId != 0)
    return;

  float *dst     = (float *) pdst;
  float *data    = (float *) pdata;
  float *weights = (float *) pweights;
  uint64_t *indices = (uint64_t *) pindices;
  uint64_t *offsets = (uint64_t *) poffsets;

  const size_t segments    = offsetsSize;
  const size_t totalLength = indicesSize;

  // NOTE : Pitch is passed as the number of elements, not as bytes.
  const size_t lineSize = dataDim1Pitch;

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

