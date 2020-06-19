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
#include <cfenv>

#include "utils.h" // From include/internal path
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibUtils.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind elKind>
inline typename std::enable_if_t<(isQuantizedElemKind(elKind)||(elKind==Float16Ty)), void> 
fwdLibAdaptiveAvgPoolInst(LibTensor* outT, LibTensor* inT, uint64_t flags,
                          const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;
  
  assert(inT->getElementType() == outT->getElementType());

  assert((inT->getElementType() == Float16Ty)||(inT->getElementType() == Int8QTy));

  size_t rounding_mode = fegetround();
  std::fesetround(FE_UPWARD);

  using elkType = typename elemKind2elemTy<elKind>::type; 

  auto inH = inT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();


#define START_IND(a, b, c) (a * c) / b
#define END_IND(a, b, c) ((a + 1) * c - 1) / b + 1

  //For each input in the batch
  for (size_t n = 0; n < outT->dims()[0]; n++) {
    //For each layer in the output tensor
    for (size_t z = 0; z < inT->dims()[3]; z++) {
      //For each value in the output tensor
      for (size_t ax = 0; ax < outT->dims()[1]; ax++) {

	unsigned int x = START_IND(ax, outT->dims()[1], inT->dims()[1]);
	unsigned int kH = END_IND(ax, outT->dims()[1], inT->dims()[1]) - x;

	for (size_t ay = 0; ay < outT->dims()[2]; ay++) {
	  unsigned int y = START_IND(ay, outT->dims()[2], inT->dims()[2]);
	  unsigned int kW = END_IND(ay, outT->dims()[2], inT->dims()[2]) - y;
	  
	  float sum = 0;

	  for (size_t fx = 0; fx < kH; fx++) {
	    for (size_t fy = 0; fy < kW; fy++) {
	      
	      dim_t ox = x + fx;
	      dim_t oy = y + fy;

	      std::array<size_t, 4> InIndices = {n, ox, oy, z};	      
	      dim_array_t extStrides = outT->strides();

	      if (elKind == Float16Ty) {
		float dst = 0;
		/*the cast avoid compilation error due to quantize types are handle together here.*/
		convertFp16ToFp32(static_cast<uint16_t>(inH.at(InIndices, extStrides, 3)), dst);
		sum += dst;
	      }
	      else
		sum += dequantize<elkType>(inH.at(InIndices, extStrides, 3),
					   inH.getScale(), 
					   inH.getOffset());
	    }
	  }
	  float kHW = kH * kW;
	  float invkHW;
	  fpReciprocalSingleElement(kHW, invkHW);
	  std::array<size_t, 4> OutIndices = {n, ax, ay, z};

	  if (elKind == Float16Ty) {
	    uint16_t dst = 0;
	    convertFp32ToFp16((sum * invkHW), dst);
	    outH.at(OutIndices) = dst;
	  }
	  else 
	    outH.at(OutIndices) = quantize<elkType>((sum * invkHW), 
						    outT->getScale(),
						    outT->getOffset());

	} // W
      } // H
    } // C
  } // N
  
  std::fesetround(rounding_mode);
  outT->evict(DO_EVICTS);

#undef START_IND
#undef END_IND
}
 
template <ElemKind elKind>
inline typename std::enable_if_t<(!isQuantizedElemKind(elKind) && (elKind != Float16Ty) && (elKind != BoolTy)), void> 
fwdLibAdaptiveAvgPoolInst(LibTensor* outT, LibTensor* inT, uint64_t flags,
                          const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) { 

  assert(inT->getElementType() == outT->getElementType());

  assert((inT->getElementType() == FloatTy)||(inT->getElementType() == Int32ITy)||
	 (inT->getElementType() == Int64ITy));

  if (get_minion_id() != minionOffset) return;

  using elkType = typename elemKind2elemTy<elKind>::type;

  auto inH = inT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();

#define START_IND(a, b, c) (a * c) / b
#define END_IND(a, b, c) ((a + 1) * c - 1) / b + 1

  //For each input in the batch
  for (size_t n = 0; n < outT->dims()[0]; n++) {
    //For each layer in the output tensor
    for (size_t z = 0; z < inT->dims()[3]; z++) {
      //For each value in the output tensor
      for (size_t ax = 0; ax < outT->dims()[1]; ax++) {

	unsigned int x = START_IND(ax, outT->dims()[1], inT->dims()[1]);
	unsigned int kH = END_IND(ax, outT->dims()[1], inT->dims()[1]) - x;

	for (size_t ay = 0; ay < outT->dims()[2]; ay++) {
	  unsigned int y = START_IND(ay, outT->dims()[2], inT->dims()[2]);
	  unsigned int kW = END_IND(ay, outT->dims()[2], inT->dims()[2]) - y;
	  
	  elkType sum = 0;

	  for (size_t fx = 0; fx < kH; fx++) {
	    for (size_t fy = 0; fy < kW; fy++) {	      
	      dim_t ox = x + fx;
	      dim_t oy = y + fy;
	      
	      std::array<size_t, 4> InIndices = {n, ox, oy, z};
	      dim_array_t extStrides = outT->strides();
	      sum += inH.at(InIndices, extStrides, 3);
	    }
	  }

 	  float kHW = kH * kW;
	  float invkHW;
	  fpReciprocalSingleElement(kHW, invkHW);
	  std::array<size_t, 4> OutIndices = {n, ax, ay, z};
	  outH.at(OutIndices) = elkType(sum * invkHW);

	} // W
      } // H
    } // C
  } // N

  outT->evict(DO_EVICTS);

#undef START_IND
#undef END_IND 
}

} // inlining

} // dnn_lib

#endif // _ADAPTIVE_AVG_POOL_INST_H_
