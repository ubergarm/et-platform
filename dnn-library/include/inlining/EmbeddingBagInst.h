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
template <ElemKind elKind>
inline typename std::enable_if_t<(elKind == Float16Ty), void>
fwdLibEmbeddingBagInst(LibTensor* outT, LibTensor *in1T, LibTensor* in2T, 
                       LibTensor* in3T, LibTensor* in4T, bool hasEndOffset,
                       uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  
  if (get_minion_id() != minionOffset) return;

  assert(in1T->getElementType() == outT->getElementType());
  assert(in1T->getElementType() == Float16Ty);
  assert((in3T->getElementType() == Int64ITy) && (in3T->getElementType() == in4T->getElementType()));

  using elkType = typename elemKind2elemTy<elKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = in1T->getHandle<elkType>();
  auto weightH = in2T->getHandle<elkType>();
  auto indxH = in3T->getHandle<int64_t>();
  auto offH = in4T->getHandle<int64_t>();

  outH.zero();

  const dim_t segments = hasEndOffset ? (in4T->dims()[0] - 1) : in4T->dims()[0];
  const dim_t numIndices = in3T->dims()[0];

/*   // NOTE : Pitch is passed as the number of elements, not as bytes. */
/*   const size_t lineSize = dataDim1Pitch; */
/*   //@TODO in SW-2429 remove dataDim1Pitch param once the instruction bellow It works.  */
/*   //const dim_t lineSize = (in1T->strides().data()[0]/in1T->getElementSize()); */
  dim_t lineSize = in1T->strides()[0];
  //dim_t lineSize = in1T->actualSize() / in1T->getElementSize();

  dim_t curIdx = 0;
  for (dim_t i = 0; i < segments; i++) {
    dim_t start = offH.raw(i);
    dim_t end;
    if(!hasEndOffset) {
      // Note that in this case we have to use numIndices to find the end of
      // the last segment. This is an issue though because it relies on knowing
      // the total length of the indices tensor which may not be possible.
      // Future implementations of this operator should always give an end
      // offset so eventually this case should be removed.
      end = (i == (segments-1))? numIndices : offH.raw(i + 1);
    }
    else {
      end = offH.raw(i + 1);
    }

    if (start == end) {
      continue;
    }
    else if (start > end) {
      break;
    }

    for (dim_t j = start; j < end; j++) {

      float weightfl;
      convertFp16ToFp32(static_cast<uint16_t>(weightH.raw(curIdx)), weightfl);
      dim_t offsetIn = indxH.raw(curIdx++) * lineSize;
      dim_t offsetOut = i * lineSize;
      for (dim_t k = 0; k < lineSize; k++) {
	float datafl = 0;
	float outfl = 0;
	uint16_t out16 = 0;
	convertFp16ToFp32(static_cast<uint16_t>(dataH.raw(offsetIn++)), datafl);
	convertFp16ToFp32(static_cast<uint16_t>(outH.raw(offsetOut)), outfl);
	outfl += datafl * weightfl;
	convertFp32ToFp16(outfl,out16);
	outH.raw(offsetOut++) = out16;
      }
    }
  }

  outT->evict(DO_EVICTS);
}

template <ElemKind elKind>
inline typename std::enable_if_t<(elKind == FloatTy), void>
fwdLibEmbeddingBagInst(LibTensor* outT, LibTensor *in1T, LibTensor* in2T, 
		       LibTensor* in3T, LibTensor* in4T, bool hasEndOffset,
		       uint64_t flags const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(in1T->getElementType() == outT->getElementType());
  assert(in1T->getElementType() == FloatTy);
  assert((in3T->getElementType() == Int64ITy) && (in3T->getElementType() == in4T->getElementType()));

  using elkType = typename elemKind2elemTy<elKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = in1T->getHandle<elkType>();
  auto weightH = in2T->getHandle<elkType>();
  auto indxH = in3T->getHandle<int64_t>();
  auto offH = in4T->getHandle<int64_t>();

  outH.zero();

  const dim_t segments = hasEndOffset ? (in4T->dims()[0] - 1) : in4T->dims()[0];
  const dim_t numIndices = in3T->dims()[0];

/*   // NOTE : Pitch is passed as the number of elements, not as bytes. */
/*   const size_t lineSize = dataDim1Pitch; */
/*   //@TODO in SW-2429 remove dataDim1Pitch param once the instruction bellow It works.  */
/*   //const dim_t lineSize = (in1T->strides().data()[0]/in1T->getElementSize()); */
  dim_t lineSize = in1T->strides()[0];
  //dim_t lineSize = in1T->actualSize() / in1T->getElementSize();

  dim_t curIdx = 0;
  for (dim_t i = 0; i < segments; i++) {
    dim_t start = offH.raw(i);
    dim_t end;
    if(!hasEndOffset) {
      // Note that in this case we have to use numIndices to find the end of
      // the last segment. This is an issue though because it relies on knowing
      // the total length of the indices tensor which may not be possible.
      // Future implementations of this operator should always give an end
      // offset so eventually this case should be removed.
      end = (i == (segments-1))? numIndices : offH.raw(i + 1);
    }
    else {
      end = offH.raw(i + 1);
    }

    if (start == end) {
      continue;
    }
    else if (start > end) {
      break;
    }

    for (dim_t j = start; j < end; j++) {
      elkType weight = weightH.raw(curIdx);      
      dim_t offsetIn = indxH.raw(curIdx++) * lineSize;
      dim_t offsetOut = i * lineSize;
      for (dim_t k = 0; k < lineSize; k++) {
	outH.raw(offsetOut++) += dataH.raw(offsetIn++) * weight;
      }
    }
  }

  outT->evict(DO_EVICTS);
}

} // namespace inlining
} // namespace dnn_lib

#endif // _EMBEDDING_BAG_INST_H_
