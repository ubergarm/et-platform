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

#include "Float16.h"
#include "LibTensor.h"
#include "utils.h"
#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

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
 * @param[out] outT LibTensor pointer to the output matrix.
 * @param[in] inT LibTensor pointer to the input matrix
 * @param[in] imm.
 */
template <typename srcType, typename opType>
INLINE_ATTR void fwdLibElementImmLogic(LibTensor* outT, LibTensor* inT, srcType imm_value,
                                       [[maybe_unused]] uint64_t flags, const uint32_t minionOffset = 0,
                                       [[maybe_unused]] const uint32_t assignedMinions = 0) {

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  void* dst = outT->getRawDataPointer();
  const srcType *aSrcT1 = inT->getRawDataPointer<srcType>();
  srcType *aDstT = outT->getRawDataPointer<srcType>();

  const dim_t* actIndex = inT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t* actPitch = inT->strides().data();

  dim_t srcDimNum = inT->ndims();

  auto numElemsDst = dstPitch[0] * actIndex[0];

  size_t initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dst);
  if (maxRead == 0)
    return;

  dim_array_t coord = {0};
  dim_t k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex, k);

  size_t offsetIn = 0;
  size_t offsetOut = 0;
  for (size_t j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  Operator<srcType, srcType, srcType, opType> op;
  size_t posMax = maxRead + initialAddr;
  bool done = false;
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n");
  while (!done && (offsetOut < posMax)) {
    op.doOp(aDstT, aSrcT1, imm_value, offsetOut, offsetIn);
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex, actPitch, dstPitch);
  }

  /* maintain compatibility through the new Iface Libtensor */
  if (!DO_EVICTS)
    return;
  size_t clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0)
    evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize * initialAddr, clperminion);
}

  // and instance for Int8Converter
template <ElemKind elK>
INLINE_ATTR void fwdLibInt8ConverterInst(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                         const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  srcType imm_value = static_cast<srcType>(0x80);
  inlining::fwdLibElementImmLogic<srcType, Xor>(outT, inT, imm_value, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _ELEMENT_IMM_LOGIC_H_
