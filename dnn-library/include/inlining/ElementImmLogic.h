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
 * @param[out] dstT Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] srcT1 Pointer to the first input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] srcPitches Vector of pitches of the first input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] scale, offset Parameters for the quantization.
 */
template <typename srcType, typename opType>
inline void fwdLibElementImmLogic(void *dstT, void *dstDims,
                                     void *dstPitches, void *srcT1,
                                     void *srcDims, void *srcPitches,
                                     unsigned int srcDimNum, void *imm,
                                     const float *scale, const int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const srcType *aSrcT1 = reinterpret_cast<srcType*>(srcT1);
  srcType *aDstT = reinterpret_cast<srcType*>(dstT);
  const srcType *imm_ptr = reinterpret_cast<srcType*>(imm);
  const srcType imm_value = *imm_ptr;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

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

} // namespace inlining

} // namespace dnn_lib

#endif // _ELEMENT_IMM_LOGIC_H_
