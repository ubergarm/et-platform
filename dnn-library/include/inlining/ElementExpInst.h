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

/**
 * @brief Given a tensor, it gives the the exponential of each element.
 *
 * Given a tensor A, it generates the output tensor B in the following way 
 * @f$ B_{i,j} = e^{A_{i,j}} @f$.
 * 
 * @tparam srcType The type of the elements in the input tensors.
 * @tparam opType An operator that takes two srcType elements and returns a 
    bool.
 * @param[out] dstT Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] srcT Pointer to the first input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] srcPitches Vector of pitches of the first input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] scale, offset Parameters for the quantization.
 */
template <typename srcType>
void dnn_lib::fwdLibElementExpInst(void *dstT, void *dstDims, void *dstPitches,
                                   void *srcT, void *srcDims, void *srcPitches,
                                   unsigned int srcDimNum, const float *scale,
                                   const int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  srcType *tOutput = (srcType *)dstT;
  srcType *tInput = (srcType *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;

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
              auto val = tInput[addrSrc];
	      float res = getExp((float)val);
              tOutput[addrDst] = res;
            }
          }
        }
      }
    }
  }
}

GEN_INSTANCES_OP(template, fwdLibElementExpInst,void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT1, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, const float * scale,
                                 const int32_t * offset);
