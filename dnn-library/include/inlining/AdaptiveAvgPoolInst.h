/*-------------------------------------------------------------------------
 * Copyright (C) 2020, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _ADAPTIVE_AVG_POOL_INST_H_
#define _ADAPTIVE_AVG_POOL_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "utils.h" // From include/internal path
#include "Addresser.h" // From include/internal path
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
inline void fwdLibAdaptiveAvgPoolInst(LibTensor* outT, LibTensor* inT,
                                      uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  if (get_minion_id() != minionOffset) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  void* srcT = inT->getRawDataPointer<void>();
  void* dstT = outT->getRawDataPointer<void>();
  
  Addresser<elK> tOutput(dstT, outT->getScale(), outT->getOffset());
  const Addresser<elK> tAInput(srcT, inT->getScale(), inT->getOffset());
  
  // unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const dim_t *dstIndex = outT->dims().data();  
  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const dim_t *actPitch = inT->strides().data();
  
#define START_IND(a, b, c) (a * c) / b
#define END_IND(a, b, c) ((a + 1) * c - 1) / b + 1

  // For each input in the batch:
  for (size_t n = 0; n < dstIndex[0]; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < actIndex[3]; z++) {
      // For each value in the output tensor::
      for (size_t ax = 0; ax < dstIndex[1]; ax++) {

        unsigned int x = START_IND(ax, dstIndex[1], actIndex[1]);
        unsigned int kH = END_IND(ax, dstIndex[1], actIndex[1]) - x;

        for (size_t ay = 0; ay < dstIndex[2]; ay++) {

          unsigned int y = START_IND(ay, dstIndex[2], actIndex[2]);
          unsigned int kW = END_IND(ay, dstIndex[2], actIndex[2]) - y;
          
          float sum = tAInput[0];
          sum = 0;

          for (size_t fx = 0; fx < kH; fx++) {
            for (size_t fy = 0; fy < kW; fy++) {
              unsigned int ox = x + fx;
              unsigned int oy = y + fy;

              sum += tAInput[n * actPitch[0] + (size_t)ox * actPitch[1] +
                             (size_t)oy * actPitch[2] + z * dstPitch[3]];
            }
          }

          float kHW = kH * kW;
          float invkHW;
          fpReciprocalSingleElement(kHW, invkHW);
          tOutput[n * dstPitch[0] + (size_t)ax * dstPitch[1] +
                  (size_t)ay * dstPitch[2] + z * dstPitch[3]] = sum * invkHW;
        } // W
      }   // H
    }     // C
  }       // N
#undef START_IND
#undef END_IND
}

} // inlining

} // dnn_lib

#endif // _ADAPTIVE_AVG_POOL_INST_H_
