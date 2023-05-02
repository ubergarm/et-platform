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

#include "LibCommon.h"
#include "LibTensor.h"
#include "LibTypes.h"
#include "utils.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Does a running accumulation of values in axis for a fixed position
 *
 * @param[out] outT LibTensor destination. It holds the expected data.
 * @param[in] inT LibTensor input. It keeps the inputs to being handle.
 * @param[in] pos Position to use as base
 * @param[in] exclusive Applied if it is true
 * @param[in] begin Beginning index
 * @param[in] end Next to last index
 * @param[in] dir Direction of accumulation
 * @param[in] axis Axis in which to accumulate
 */
template <ElemKind elKind>
INLINE_ATTR typename std::enable_if_t<(elKind == Float16Ty), void>
singleDimCumSum(LibTensor* outT, LibTensor* inT, dim_array_t pos, bool exclusive, sdim_t begin, sdim_t end, sdim_t dir,
                dim_t axis) {
  using elkType = typename elemKind2elemTy<elKind>::type;
  auto inH = inT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();

  float accum = 0;

  for (sdim_t i = begin; i != end; i += dir) {
    pos[axis] = static_cast<size_t>(i);
    if (!exclusive) {
      float dst = 0;
      convertFp16ToFp32(static_cast<uint16_t>(inH.at(pos)), dst);
      accum += dst;
    }

    uint16_t dst_tmp = 0;
    convertFp32ToFp16(accum, dst_tmp);
    outH.at(pos) = dst_tmp;

    if (exclusive) {
      float dst1 = 0;
      convertFp16ToFp32(static_cast<uint16_t>(inH.at(pos)), dst1);
      accum += dst1;
    }
  }
}

template <ElemKind elKind>
INLINE_ATTR typename std::enable_if_t<(elKind != Float16Ty), void>
singleDimCumSum(LibTensor* outT, LibTensor* inT, dim_array_t pos, bool exclusive, sdim_t begin, sdim_t end, sdim_t dir,
                dim_t axis) {
  using elkType = typename elemKind2elemTy<elKind>::type;
  auto inH = inT->getHandle<elkType>();
  auto outH = outT->getHandle<elkType>();
  elkType accum = 0;

  for (sdim_t i = begin; i != end; i += dir) {
    pos[axis] = static_cast<size_t>(i);
    if (!exclusive) {
      accum += inH.at(pos);
    }

    outH.at(pos) = accum;

    if (exclusive) {
      accum += inH.at(pos);
    }
  }
}

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
 * @param[in] axis Dimension in which to accumulate.
 * @param[flags] flags Gives the information of the Active Shires and the
 *  type of evict required.
 */
template <ElemKind elKind>
INLINE_ATTR void fwdLibCumSumInst(LibTensor* outT, LibTensor* inT, bool exclusive, bool reverse, dim_t axis,
                                  uint64_t flags, const uint32_t minionOffset = 0,
                                  [[maybe_unused]] const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  // TODO : support negative axis (need to change API)
  // if (axis < 0) {
  //   axis += inT->ndims();
  // }

  assert(inT->getElementType() == outT->getElementType());

  sdim_t begin = 0;
  sdim_t end = inT->dims()[axis];
  sdim_t dir = 1;

  if (reverse) {
    begin = end - 1;
    end = -1;
    dir = -1;
  }

  dim_array_t pos = {0, 0, 0, 0, 0, 0};
  dim_array_t maxVal = inT->dims();
  maxVal[axis] = 1;

  for (pos[0] = 0; pos[0] < maxVal[0]; ++pos[0]) {
    for (pos[1] = 0; pos[1] < maxVal[1]; ++pos[1]) {
      for (pos[2] = 0; pos[2] < maxVal[2]; ++pos[2]) {
        for (pos[3] = 0; pos[3] < maxVal[3]; ++pos[3]) {
          for (pos[4] = 0; pos[4] < maxVal[4]; ++pos[4]) {
            for (pos[5] = 0; pos[5] < maxVal[5]; ++pos[5]) {
              singleDimCumSum<elKind>(outT, inT, pos, exclusive, begin, end, dir, axis);
            }
          }
        }
      }
    }
  }

  outT->evict(DO_EVICTS);
}
}
}

#endif
