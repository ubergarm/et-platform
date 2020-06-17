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
#include "utils.h" // From include/internal path
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Convert a length vector to a range sequence. 
 *
 * For example, input=[4,3,1], the output would be [0,1,2,3,0,1,2,0].
 *
 * Currently It only solves Int32ITy ElemKind following InstGen.cpp
 * Interpreter.cpp and isOpSupported at ETSOC.cpp specification.
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected data.
 * @param[in] in1T LibTensor input. It keeps the Data to being handle.
 * @param[in] in2T LibTensor input. It keeps the Weights to being handle.
 * @param[in] in3T LibTensor input. It keeps the indices to being handle.
 * @param[in] in4T LibTensor input. It keeps the offsets to being handle.
 * @param[in] hasEndOffset bool type mark the end of the last segment.
 * @param[flags] flags Gives the information of the Active Shires and the
 *  type of evict required.
 */
template <ElemKind elKind, ElemKind accumKind>
inline typename std::enable_if_t<(elKind == Float16Ty), void>
fwdLibEmbeddingBagByteRowwiseOffsetsInst(LibTensor* outT, LibTensor *in1T, LibTensor* in2T, 
		       LibTensor* in3T, LibTensor* in4T, bool hasEndOffset,
		       uint64_t flags) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0) //if (minionId != minionOffset)
    return;

  assert(((accumKind == Float16Ty)||(accumKind == FloatTy)));

}

template <ElemKind elKind, ElemKind accumKind>
inline typename std::enable_if_t<(elKind == FloatTy), void>
fwdLibEmbeddingBagInst(LibTensor* outT, LibTensor *in1T, LibTensor* in2T, 
		       LibTensor* in3T, LibTensor* in4T, bool hasEndOffset,
		       uint64_t flags) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0) //if (minionId != minionOffset)
    return;

  assert(accumKind == FloatTy);

  using elkType = typename elemKind2elemTy<elKind>::type;

  auto indxH = in3T->getHandle<int64_t>();
  auto offH = in4T->getHandle<int64_t>();

  outH.zero();

  // If an end offset is present to mark the end of the last segment then this
  // must be subtracted to get the correct number of segments
  size_t segments = hasEndOffset ? in4T->dims()[0] - 1 : in4T->dims()[0];
  dim_t numIndices = in3T->dims()[0];

  const bool using4BitQtzt = (in1T->getElementType() == UInt4FusedFP16QTy);

  const size_t outLineSize = outT->strides()[0];

  auto outH = outT->getHandle<elkType>();
  auto dataH = in1T->getHandle<uint8_t>();
  auto weightH = in2T->getHandle<elkType>();
  
  for (dim_t i = 0; i < segments; i++) {
    std::array<accumKind> accum(outLineSize, 0.0f);
    size_t start = offH.raw(i);
    size_t end;
    if (!hasEndOffset) {
      // Note that in this case we have to use numIndices to find the end of
      // the last segment. This is an issue though because it relies on knowing
      // the total length of the indices tensor which may not be possible.
      // Future implementations of this operator should always give an end
      // offset so eventually this case should be removed.

      end = (i == (segments - 1))? numIndices: offH.raw(i+1);
    }
    else {
      end = offH.raw(i+1);
    }

    if (start == end) {
      continue;
    } else if (start > end) {
      break;
    }
    
    for (dim_t j = start; j < end; j++) {
      const float weight = static_cast<float>(weightH.raw(j));
      const size_t rowIdx = indxH.raw(j);
      elKind scale, offset;
      std::tie(scale, offset) = dataH.getFusedScaleOffsetFromRow<elKind>(rwoIdx);
      for(size_t k = 0; k < outLineSize; k++) {
	float d = 0.0f;
	if (!using4BitQtzt) {
	  d = dequantize(dataH.at(std::array<size_t,2>{rowIdx, k}), dataH.getScale(), dataH.getOffset());
	}
	else {
	  const bool isMSB = ((k%2) ==1);
	  d = dequantize4Bits(dataH.at(std::array<size_t,2>{rowIdx,(k/2)}), dataH.getScale(), dataH.getOffset(), isMSB);
	}
	accum[k] += d * weight;
      }
    }
    // Accumulation in FP32 complete, now copy back to output with cast to T.
    dim_t offsetOut = i * outLineSize;
    for (dim_t k = 0; k < outLineSize; k++) {
      outH.raw(offsetOut++) = static_cast<elkind>(accum[k]);
    }
  }
}


} // inlining
} // dnn_lib
