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

#ifndef _ELEMENT_EXP_INST_H_
#define _ELEMENT_EXP_INST_H_

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

/**
 * @brief Given a tensor, it gives the the exponential of each element.
 *
 * Given a tensor A, it generates the output tensor B in the following way 
 * @f$ B_{i,j} = e^{A_{i,j}} @f$.
 * 
 * @tparam srcType The type of the elements in the input tensors.
 * @tparam opType An operator that takes two srcType elements and returns a 
    bool.
 * @param[out] outT LibTensor pointer to the destination matrix
 * @param[in]  inT LibTensor pointer to the input Matrix
 */
template <ElemKind elK>
inline void fwdLibElementExpInst(LibTensor* outT, LibTensor* inT) {
  using srcType = typename elemKind2elemTy<elK>::type;
  
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */    
  void* srcT = inT->getRawDataPointer<void>();
  void* dstT = outT->getRawDataPointer<void>();
 
  // const Addresser<srcType> tInput(srcT, scale[0], offset[0]);  
  const Addresser<srcType> tInput(srcT, inT->getScale(), inT->getOffset());
  // Addresser<srcType> tOutput(dstT, scale[1], offset[1]);  
  Addresser<srcType> tOutput(dstT, outT->getScale(), outT->getOffset());

  // unsigned int *srcIndex = (unsigned int *)srcDims;
  const dim_t *srcIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = inT->strides().data();
  
  unsigned int srcDimNum =  static_cast<unsigned int>(inT->ndims());

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n");
  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];

              float res = static_cast<float>(M_LOG2E) * tInput[addrSrc];
              __asm__ __volatile__ ("fexp.ps %0, %0\n" : "+&f" (res) );
              tOutput[addrDst] = res;
            }
          }
        }
      }
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ELEMENT_EXP_INST_H_

