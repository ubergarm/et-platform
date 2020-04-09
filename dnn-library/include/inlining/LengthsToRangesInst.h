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

#ifndef _LENGTHS_TO_RANGES_INST_H_
#define _LENGTHS_TO_RANGES_INST_H_

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

namespace dnn_lib {

namespace inlining {

template <typename srcType>
inline void fwdLibLengthsToRangesInst(void *dstT, void *dstDims,
                                        void *dstPitches, void *plengths,
                                        unsigned int lenDim, const float *scale,
                                        const int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> lengths(plengths, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);

  unsigned int *dstPitch = (unsigned int *)dstPitches;

  auto toffset = lengths[0];
  toffset = 0;
  for (size_t i = 0; i < lenDim; i++) {
    auto length = lengths[i];
    tOutput[i * dstPitch[0]] = toffset;
    tOutput[i * dstPitch[0] + 1] = length;
    toffset += length;
  }
}

template <typename srcType>
void fwdLibLengthsToRangesInstThreaded(void *dstT, void *dstDims,
                                        void *dstPitches, void *plengths,
                                        unsigned int lenDim, const float *scale,
                                        const int32_t *offset, uint64_t flags) {

  const Addresser<srcType> lengths(plengths, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32;
  if (minionId >= activeMinions) return;
  int level = -1;
  for (unsigned int j = 1; j < activeMinions; j*=2)
    level++;

  unsigned int *dstPitch = (unsigned int *)dstPitches;

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getReversedCachelinePartition(typeSize, lenDim, initialAddr, maxRead,
                               activeMinions);

  unsigned int posMax = maxRead + initialAddr;

  float toffset = lengths[0];
  toffset = 0;
  for (size_t i = initialAddr; i < posMax; i++)
    toffset += lengths[i];

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
       "fmv.s.x f0, %[toffset]\n"
       "fand.pi f31, f0, f0\n"

       REDUCE_AND_COPY(0, 1)
       REDUCE_AND_COPY(1, 2)
       REDUCE_AND_COPY(2, 3)
       REDUCE_AND_COPY(3, 4)
       REDUCE_AND_COPY(4, 5)

       ADD_AND_BROADCAST(3, 4)
       ADD_AND_BROADCAST(2, 3)
       ADD_AND_BROADCAST(1, 2)
       ADD_AND_BROADCAST(0, 1)

       "fsub.ps f0, f0, f31\n"
       "fmv.x.s %[toffset], f0\n"

     : [toffset] "+r" (toffset)
     : [csr_red0] "r" (tensor_reduce_params(0)),
       [csr_red1] "r" (tensor_reduce_params(1)),
       [csr_red2] "r" (tensor_reduce_params(2)),
       [csr_red3] "r" (tensor_reduce_params(3)),
       [csr_red4] "r" (tensor_reduce_params(4)),
       [csr_bdc0] "r" (tensor_broadcast_params(0)),
       [csr_bdc1] "r" (tensor_broadcast_params(1)),
       [csr_bdc2] "r" (tensor_broadcast_params(2)),
       [csr_bdc3] "r" (tensor_broadcast_params(3))
     : "f0", "f1", "f2", "f3", "f4", "f5", "f31"
   );

  for (size_t i = initialAddr; i < posMax; i++) {
    float length = lengths[i];
    tOutput[i * dstPitch[0]] = toffset;
    tOutput[i * dstPitch[0] + 1] = length;
    toffset += length;
  }

#undef tensor_reduce_params
#undef tensor_broadcast_params
#undef COPY_AND_REDUCE
#undef ADD_AND_BROADCAST
}

} // namespace inlining

} // namespace dnn_lib

#endif // _LENGTHS_TO_RANGES_INST_H_
