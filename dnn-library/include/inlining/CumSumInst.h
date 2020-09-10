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

#ifndef _CUMSUM_H_
#define _CUMSUM_H_

#include <assert.h>

#include "utils.h"
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Does a running accumulation of all values in input (inclusive).
 *
 * For example, input=[4,3,1], the output would be [0,1,2,3,0,1,2,0].
 *
 * Currently restricted to FloatTy, Float16Ty, Int32ITy and Int64ITy,
 * It's follows Interpreter.cpp restrictions.
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected data.
 * @param[in] inT LibTensor input. It keeps the inputs to being handle.
 * @param[in] exclusive Applied if it is true.
 * @param[in] reverse applied if it is true.
 * @param[flags] flags Gives the information of the Active Shires and the
 *  type of evict required.
 */
template <ElemKind elKind>
inline typename std::enable_if_t<(isQuantizedElemKind(elKind)
          ||(elKind==Float16Ty)), void>
fwdLibCumSumInst(LibTensor* outT, LibTensor* inT, bool exclusive, bool reverse, 
     uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(inT->getElementType() == outT->getElementType());
  using elkType = typename elemKind2elemTy<elKind>::type;

  auto inH = inT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();

  elkType accum = 0;
  
  sdim_t s = 0;
  sdim_t n = outH.size();
  sdim_t dir = 1;

  if (reverse) {
    s = n-1;
    n = -1;
    dir = -1;
  }

  for (sdim_t i = s; i != n; i+=dir) {
    std::array<dim_t, 1> iNdx = {static_cast<size_t>(i)};
    if (!exclusive) {
      if (elKind == Float16Ty) {
  float dst = 0;
  convertFp16ToFp32(static_cast<uint16_t>(inH.at(iNdx)), dst);
  accum += dst;
      }
      else {
  accum += dequantize<elkType>(inH.at(iNdx), inT->getScale(), inT->getOffset());
      }
    }
    
    if (elKind == Float16Ty) {
      uint16_t dst = 0;
      convertFp32ToFp16(accum, dst);
      outH.at(iNdx) = dst;
    }
    else {
      outH.at(iNdx) = quantize<elkType>(accum, outT->getScale(), outT->getOffset());
    }
    
    if (exclusive) {
      if (elKind == Float16Ty) {
  float dst = 0;
  convertFp16ToFp32(static_cast<uint16_t>(inH.at(iNdx)), dst);
  accum += dst;
      }
      else {
  accum += dequantize<elkType>(inH.at(iNdx), inT->getScale(), inT->getOffset());
      }
    }

  }

  outT->evict(DO_EVICTS);
}

template <ElemKind elKind>
inline typename std::enable_if_t<(!isQuantizedElemKind(elKind) && 
          (elKind!=Float16Ty) &&
          (elKind!=BoolTy)), void>
fwdLibCumSumInst(LibTensor* outT, LibTensor* inT, bool exclusive, bool reverse, 
     uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(inT->getElementType() == outT->getElementType());
  using elkType = typename elemKind2elemTy<elKind>::type;

  auto inH = inT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();

  elkType accum = 0;
  
  sdim_t s = 0;
  sdim_t n = outH.size();
  sdim_t dir = 1;

  if (reverse) {
    s = n-1;
    n = -1;
    dir = -1;
  }

  for (sdim_t i = s; i != n; i+=dir) {
    std::array<size_t, 1> iNdx = {static_cast<size_t>(i)};
    if (!exclusive) {
      accum += inH.at(iNdx);
    }
    outH.at(iNdx) = accum;
    if (exclusive) {
      accum += inH.at(iNdx);
    }
  }
  outT->evict(DO_EVICTS);
}

}
}

#endif
