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

#ifndef _ELEMENT_IMM_LOGIC_H_
#define _ELEMENT_IMM_LOGIC_H_

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
 * @brief Immediate logic operations, like andi, ori, xori...
 *
 * Given a tensor A of intergers and an immediate (also integer) it
 * applies to each element of A the logic operation using the immediate
 * as the second source
 * 
 * @tparam srcType The type of the elements in the input tensor and immediate
 * @tparam opType An operator that takes one srcType and one immediate value 
 *  and returns a srcType (&, |, ^, etc).
 * @param[out] outT LibTensor pointer to the output matrix.
 * @param[in] inT LibTensor pointer to the input matrix
 * @param[in] imm.
 */
template <typename srcType, typename opType>
inline void fwdLibElementImmLogic(LibTensor* outT, LibTensor* inT, srcType imm_value) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */    

  // const srcType *aSrcT1 = reinterpret_cast<srcType*>(srcT1);
  const srcType *aSrcT1 = inT->getRawDataPointer<srcType>();
  // srcType *aDstT = reinterpret_cast<srcType*>(dstT);
  srcType *aDstT = outT->getRawDataPointer<srcType>();

  // unsigned int *srcIndex = (unsigned int *)srcDims;
  const dim_t *srcIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  const dim_t *srcPitch = inT->strides().data();

   unsigned int srcDimNum =  static_cast<unsigned int>(inT->ndims());

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc1, addrDst;
  Operator<srcType, srcType, srcType, opType> op;

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc1 = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                          w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                          w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              op.doOp(aDstT, aSrcT1, imm_value, addrDst, addrSrc1);
            }
          }
        }
      }
    }
  }
}

  // and instance for Int8Converter
template <typename srcType>
inline void fwdLibElementInt8Converter(LibTensor* outT, LibTensor* inT, ) {
  srcType imm_value = 0x80;
  fwdLibElementImmLogic<srcType, Xor>(outT, inT, imm_value);
}
  
} // namespace inlining

} // namespace dnn_lib

#endif // _ELEMENT_IMM_LOGIC_H_
