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

#ifndef _BATCHED_REDUCE_MIN_H_
#define _BATCHED_REDUCE_MIN_H_

#include <limits>
#include <assert.h>

#include "utils.h"
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Performs Reduce Min operation on the Input given Axes. 
 *        based on the input \p batch type with dimensions specified with \p axes
 *
 * Currently It only supports FloatTy, Float16Ty, Int32Ity, In64ITy ElemKinds.
 * Following InstGen.cpp Interpreter.cpp and isOpSupported at ETSOC.cpp specification.
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected.
 * @param[in] inT LibTensor input. It keeps the inputs to being handle.
 * @param[flags] flags Gives the information of the Active Shires and the
 *  type of evict required.
 */
template <ElemKind elKind, size_t N>
inline typename std::enable_if_t<(isQuantizedElemKind(elKind)||(elKind==Float16Ty)), void>
fwdLibBatchedReduceMinInst(LibTensor* outT, LibTensor* inT,
         const std::array<uint32_t, N> &axes, 
         uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;
  
  /*@TODO NOT TESTED.*/
  assert(inT->getElementType() == outT->getElementType());
  assert(inT->getElementType() == Float16Ty);

  using elkType = typename elemKind2elemTy<elKind>::type; 

  auto outH = outT->getHandle<elkType>();

  elkType max = std::numeric_limits<elkType>::max();

  outH.clear(max);

  /* Copy of input strides and set to 0 the strides which Dimension we want to reduce.*/
  /* Rewrite with output strides over this jumping the stride dimension to be reduced. */
  dim_array_t dstPitch = inT->strides();
  for (size_t i = 0; i< axes.size(); i++)
    dstPitch[axes[i]] = 0;
  for (size_t i = 0, j = 0; i < dstPitch.size(); i++) {
    if (dstPitch[i] !=0) {
      dstPitch[i] = outT->strides()[j++];
    }
  }

  elkType* outRawData = outT->getRawDataPointer<elkType>();
  elkType* inRawData = inT->getRawDataPointer<elkType>();


  /* At dims_loop 1st param is where the loop acts (tipically one of the tensors), */
  /* 2nd and 3th. param are the strides which generate the input params (at the same order) */
  /* of func/lambda needs to work */
  dims_loop<>::run(inT->dims(), inT->strides(), dstPitch,
       [&](size_t addrSrc, size_t addrDst) {

         float dst, dst2, value = 0;         
         if (elKind == Float16Ty) {
           convertFp16ToFp32(static_cast<uint16_t>(inRawData[addrSrc]), dst);
           convertFp16ToFp32(static_cast<uint16_t>(outRawData[addrDst]), dst2);          
         }
         else {
           dst = dequantize<elkType>(inRawData[addrSrc], inT->getScale(), inT->getOffset());
           dst2 = dequantize<elkType>(outRawData[addrSrc], outT->getScale(), outT->getOffset());         
         }       

         value = (dst2 < dst)? dst2 : dst;

         if (elKind == Float16Ty) {
           uint16_t valint = 0;
           convertFp32ToFp16(value, valint);
           outRawData[addrDst] = valint;
         }
         else {
           outRawData[addrDst] = quantize<elkType>(value, outT->getScale(), outT->getOffset());
         }
       });

  outT->evict(DO_EVICTS);
}

template <ElemKind elKind, size_t N>
inline typename std::enable_if_t<(!isQuantizedElemKind(elKind) && 
          (elKind != Float16Ty) && (elKind != BoolTy)), void>
fwdLibBatchedReduceMinInst(LibTensor* outT, LibTensor* inT, 
         const std::array<uint32_t, N> &axes, 
         uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(inT->getElementType() == outT->getElementType());

  using elkType = typename elemKind2elemTy<elKind>::type; 

  auto outH = outT->getHandle<elkType>();

  elkType max = std::numeric_limits<elkType>::max();

  outH.clear(max);

  /* Copy of input strides and set to 0 the strides which Dimension we want to reduce.*/
  /* Rewrite with output strides over this jumping the stride dimension to be reduced. */
  dim_array_t dstPitch = inT->strides();
  for (size_t i = 0; i< axes.size(); i++)
    dstPitch[axes[i]] = 0;
  for (size_t i = 0, j = 0; i < dstPitch.size(); i++) {
    if (dstPitch[i] !=0) {
      dstPitch[i] = outT->strides()[j++];
    }
  }

  elkType* outRawData = outT->getRawDataPointer<elkType>();
  elkType* inRawData = inT->getRawDataPointer<elkType>();

  /* At dims_loop 1st param is where the loop acts (tipically one of the tensors), */
  /* 2nd and 3th. param are the strides which generate the input params (at the same order) */
  /* of func/lambda needs to work */
  dims_loop<>::run(inT->dims(), inT->strides(), dstPitch,
       [&](size_t addrSrc, size_t addrDst) {
         outRawData[addrDst] = (outRawData[addrDst] < inRawData[addrSrc])?
                             outRawData[addrDst] : inRawData[addrSrc];
       });

  outT->evict(DO_EVICTS);
}

} //inlining

} //dnn_lib 

#endif // 
