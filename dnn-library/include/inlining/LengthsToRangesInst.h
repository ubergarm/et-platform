/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _LENGTHS_TO_RANGES_INST_H_
#define _LENGTHS_TO_RANGES_INST_H_

#include <assert.h>
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>

#include "Addresser.h"
#include "Float16.h"
#include "LibTensor.h"
#include "utils.h"

namespace dnn_lib {

namespace inlining {

template <ElemKind elK>
void fwdLibLengthsToRangesInst(LibTensor* outT, LibTensor* inT, [[maybe_unused]] uint64_t flags,
                               const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  assert(get_minion_id() >= minionOffset);
  size_t minionId = get_minion_id() - minionOffset;
  size_t activeMinions = (assignedMinions == 0) ? 32 : assignedMinions;
  if (activeMinions > 32) activeMinions = 32;
  if (minionId >= activeMinions) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer();
  void* plengths = inT->getRawDataPointer();

  const Addresser<elK> lengths(plengths, inT->getScale(), inT->getOffset());
  Addresser<elK>    tOutput(dstT, outT->getScale(), outT->getOffset());

  ssize_t level = -1;
  for (size_t j = 1; j < activeMinions; j *= 2)
    level++;

  const dim_t *dstPitch = outT->strides().data();
  
  const dim_t *lenIndx = inT->dims().data();

  size_t initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getReversedCachelinePartition(typeSize, lenIndx[0], initialAddr, maxRead,
                               activeMinions);

  size_t posMax = maxRead + initialAddr;

  float offset = static_cast<float>(lengths[0]);
  offset = 0;
  for (size_t i = initialAddr; i < posMax; i++) {
    offset += static_cast<float>(lengths[i]);
  }

#define tensor_reduce_params(_lvl)                    \
        ((0ULL  & 0x2)        << 62) |                \
        ((0ULL  & 0x1F)       << 57) |                \
        ((0ULL  & 0x1FFFFFFF) << 28) |                \
        ((0ULL  & 0xF)        << 24) |                \
        ((1ULL  & 0xFF)       << 16) |                \
        ((_lvl  & 0x1FFF)     << 3)  |                \
        ((0ULL  & 0x1)        << 2)  |                \
        ((0x3   & 0x3))

#define tensor_broadcast_params(_lvl)                 \
        ((0ULL  & 0x2)        << 62) |                \
        ((0ULL  & 0x1F)       << 57) |                \
        ((0ULL  & 0x1FFFFFFF) << 28) |                \
        ((0ULL  & 0xF)        << 24) |                \
        ((1ULL  & 0xFF)       << 16) |                \
        ((_lvl  & 0x1FFF)     << 3)  |                \
        ((0ULL  & 0x1)        << 2)  |                \
        ((0x2   & 0x3))

#define REDUCE_AND_COPY(_J, _FD)                      \
        "csrw 0x800, %[csr_red" #_J "]\n"             \
        "fand.pi f" #_FD ", f0, f0\n"

#define ADD_AND_BROADCAST(_J, _FD)                    \
        "fsub.ps f0, f0, f" #_FD "\n"                 \
        "csrw tensor_reduce, %[csr_bdc" #_J "]\n"     \
        "fadd.ps f0, f0, f" #_FD "\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0x1\n"
    "fmv.s.x f0, %[offset]\n"
    "fand.pi f31, f0, f0\n"

    REDUCE_AND_COPY(0, 1) REDUCE_AND_COPY(1, 2) REDUCE_AND_COPY(2, 3) REDUCE_AND_COPY(3, 4) REDUCE_AND_COPY(4, 5)

      ADD_AND_BROADCAST(3, 4) ADD_AND_BROADCAST(2, 3) ADD_AND_BROADCAST(1, 2) ADD_AND_BROADCAST(0, 1)

        "fsub.ps f0, f0, f31\n"
        "fmv.x.s %[offset], f0\n"

    : [ offset ] "+r"(offset)
    : [ csr_red0 ] "r"(tensor_reduce_params(0)), [ csr_red1 ] "r"(tensor_reduce_params(1)),
      [ csr_red2 ] "r"(tensor_reduce_params(2)), [ csr_red3 ] "r"(tensor_reduce_params(3)),
      [ csr_red4 ] "r"(tensor_reduce_params(4)), [ csr_bdc0 ] "r"(tensor_broadcast_params(0)),
      [ csr_bdc1 ] "r"(tensor_broadcast_params(1)), [ csr_bdc2 ] "r"(tensor_broadcast_params(2)),
      [ csr_bdc3 ] "r"(tensor_broadcast_params(3))
    : "f0", "f1", "f2", "f3", "f4", "f5", "f31");

  static_assert(elK == Int32ITy, "Unsupported elK type.");
  for (size_t i = initialAddr; i < posMax; i++) {
    float length = static_cast<float>(lengths[i]);
    tOutput[i * dstPitch[0]] = static_cast<srcType>(offset);
    tOutput[i * dstPitch[0] + 1] = static_cast<srcType>(length);
    offset += length;
  }

#undef tensor_reduce_params
#undef tensor_broadcast_params
#undef COPY_AND_REDUCE
#undef ADD_AND_BROADCAST
}

} // namespace inlining

} // namespace dnn_lib

#endif // _LENGTHS_TO_RANGES_INST_H_
