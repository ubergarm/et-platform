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

#ifndef _SPACE_TO_DEPTH_H_
#define _SPACE_TO_DEPTH_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>
#include <cfenv>

#include "utils.h" // From include/internal path
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibUtils.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Rearranges blocks of spatial data, into depth. More specifically, 
 * this op outputs a copy of the input tensor where values from the height 
 * and width dimensions are moved to the depth dimension. The attr 
 * block_size indicates the input block size.
 *
 * Only support for FloatTy Float16Ty Int8QTy Int64ITy  
 * BoolTy and Fused kinds not supported
 * Following InstGen.cpp Interpreter.cpp and isOpSupported at ETSOC.cpp specification.
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected.
 * @param[in] inT LibTensor input. It keeps the inputs to being handle.
 * @param[in] blockSize the input block size
 * @param[flags] flags Gives the information of the Active Shires and the
 * type of evict required.
 */ 
template <ElemKind elKind>
inline void fwdLibSpaceToDepthInst(LibTensor* outT, LibTensor* inT, 
           const uint32_t blockSize, uint64_t flags, 
           const uint32_t minionOffset = 0,
           const uint32_t assignedMinions = 0) { 

  if (get_minion_id() != minionOffset) return;

  assert(inT->getElementType() == outT->getElementType());
  assert(((inT->getElementType() == FloatTy)||(inT->getElementType() == Int64ITy)));

  using elkType = typename elemKind2elemTy<elKind>::type;

  auto inH = inT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();

  size_t inDepth = inT->dims()[3];

  for (size_t ob = 0; ob < outT->dims()[0]; ++ob) {
    for (size_t oh = 0; oh < outT->dims()[1]; ++oh) {
      for (size_t ow = 0; ow < outT->dims()[2]; ++ow) {
        for (size_t oc = 0; oc < outT->dims()[3]; ++oc) {
          // Gets the block layer we are on
          size_t blockDepthLayer = oc / inDepth;
          // every multiple of block size we reset to 0 offset
          size_t iw = ow * blockSize + blockDepthLayer % blockSize;
          // every multiple of blockSize we start height traversal + 1
          size_t ih = oh * blockSize + blockDepthLayer / blockSize;
          // at every multiple of inDepth index in to input depths resets to 0
          size_t ic = oc % inDepth;

          outH.at(std::array<size_t,4>{ob, oh, ow, oc}) = inH.at(std::array<size_t,4>{ob, ih, iw, ic});
        }
      }
    }
  }
}

} // inlining
} // dnn_lib

#endif
