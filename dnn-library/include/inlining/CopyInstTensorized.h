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

#ifndef _COPY_INST_TENSORIZED_H_
#define _COPY_INST_TENSORIZED_H_

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
 * @brief Copies the src matrix to the dst matrix.
 *
 * It makes a copy of the tensor src into the dst tensor, which may not
 * have the same pitches or dimensions, so it allows a reshaping. This is
 * the threaded and vectorized version for this operator.
 * 
 * @warning It is assumed that the destination tensor starts at the beginning
 *  of a cacheline.
 * 
 * @warning It is assumed that the input and output tensors have the same shape
 *  (same dimensions and pitches).
 *
 * @tparam srcType The type of the elements in the tensor.
 * @param[out] dst Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] src Pointer to the input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] srcPitches Vector of pitches of the input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] scale, offset Parameters for the quantization.
 * @param[in] flags Gives the information of the Active Shires and the
 *  type of evict required.
 * @param[in] minionOffset The first minion that is assigned to this node.
 * @param[in] assignedMinions Amount of minions avaliable.
 */
template <typename srcType>
inline void fwdLibCopyInstTensorized(LibTensor* inT, LibTensor* outT,
                                     uint64_t flags,
                                     const uint32_t minionOffset = 0,
                                     const uint32_t assignedMinions = 0) {

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if ((minionId >= activeMinions) || (minionId >= activeMinions))
    return;

  /* maintain compatibility through the new Iface Libtensor */

  auto srcH = inT->getHandle<srcType>();
  auto dstH = outT->getHandle<srcType>();
  void* src = reinterpret_cast<void*>(srcH.getUnsafePtrdbg());
  void* dst = reinterpret_cast<void*>(dstH.getUnsafePtrdbg());
 
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  dim_t dstIndex[max_tensor_dimensions] = {0,};
  dstH.cpydims(dstIndex);
  // unsigned int *actIndex = (unsigned int *)srcDims;
  dim_t actIndex[max_tensor_dimensions] = {0,};
  srcH.cpydims(actIndex);

  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  dim_t dstPitch[max_tensor_dimensions] = {0,};
  dstH.cpypitchesdbg(dstPitch);
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  dim_t actPitch[max_tensor_dimensions] =  {0,};
  srcH.cpypitchesdbg(actPitch);

  size_t typeSize = getsize<srcType>();
  uint64_t numElemsDst = dstPitch[0] * actIndex[0] *
                             typeSize; // Total number of elements in the tensor
  uint64_t numCacheLines = (numElemsDst - 1) / CACHE_LINE_BYTES + 1; //64 = CacheLineLength
  uint64_t minionCacheLines = (numCacheLines - 1) / activeMinions + 1;
  uint64_t initialCacheLine = minionCacheLines * minionId;
  uint64_t lastCacheLine = initialCacheLine + minionCacheLines;
  minionCacheLines =
          (lastCacheLine <= numCacheLines) ? minionCacheLines
        : (initialCacheLine < numCacheLines) ? numCacheLines - initialCacheLine : 0;
  uint64_t srcAddr = (uint64_t)src + initialCacheLine*CACHE_LINE_BYTES;
  uint64_t dstAddr = (uint64_t)dst + initialCacheLine*CACHE_LINE_BYTES;

  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
  while (minionCacheLines >= 16) {
    tensor_load(0, 0, 0, 0, 0, srcAddr, 0, 0xF, 0x40, 0);
    WAIT_TENSOR_LOAD_0;
    srcAddr += 1024;
    minionCacheLines -= 16;
    tensor_store_scp(0, 0, 0xF, dstAddr, 0x40);
    dstAddr += 1024;
  }
  if (minionCacheLines == 0) return;

  tensor_load(0, 0, 0, 0, 0, srcAddr, 0, minionCacheLines-1, 0x40, 0);
  WAIT_TENSOR_LOAD_0;
  tensor_store_scp(0, 0, minionCacheLines-1, dstAddr, 0x40);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _COPY_INST_TENSORIZED_H_
