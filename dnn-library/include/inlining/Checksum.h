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

#ifndef _CHECKSUM_H_
#define _CHECKSUM_H_

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
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

inline void uint32_to_ascii_hex(char *s, uint32_t value) {
  for (uint32_t i = 0; i < 8; i++) {
    uint32_t bits = (value >> (28 - i * 4)) & 0xF;
    s[i] = (bits <= 9) ? ('0' + bits) : ('A' + bits - 10);
  }
}

template <typename srcType>
inline void fwdLibChecksum(LibTensor* inT, uint64_t flags) {
  // The checksum is the u32 addition of all the non-padding bytes of the tensor
  uint32_t checksum = 0;

  void* src = inT->getRawDataPointer<void>();
  
  // unsigned int *actIndex = (unsigned int *)srcDims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *actPitch = (unsigned int *)srcPitches;
  const dim_t *actPitch = inT->strides().data();

  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  size_t typeSize = getsize<srcType>();
  unsigned int srcNumElems = actIndex[0] * actPitch[0];

  unsigned int minionElems = srcNumElems / activeMinions;
  unsigned int mod = srcNumElems - minionElems * activeMinions;
  unsigned int startPos;
  if (minionId < mod) {
    ++minionElems;
    startPos = minionElems * minionId;
  } else
    startPos = mod + minionId * minionElems;

  if (minionElems != 0) {
    unsigned int endPos = startPos + minionElems;

    unsigned int coordIn[srcDimNum];
    unsigned int lastNonZeroCoord;
    getNonPaddingCoordinates(coordIn, startPos, srcDimNum, actPitch, actIndex,
                             lastNonZeroCoord);

    unsigned int offsetIn = 0;
    for (unsigned int i = 0; i < lastNonZeroCoord; i++)
      offsetIn += actPitch[i] * coordIn[i];

    bool done = false;
    while (!done && (offsetIn < endPos)) {
      uint8_t *elemAddr = (uint8_t *)src + offsetIn * typeSize;
      // Iterate over all the bytes of the element
      for (size_t i = 0; i < typeSize; i++) {
        checksum += (uint32_t)*(elemAddr + i);
      }
      done = getOffsets(srcDimNum, coordIn, offsetIn, actIndex, actPitch);
    }
  }

  // Reduce CheckSum across active minions
  // TODO: make this general for non power of two active shires
  size_t level = 4;
  for (size_t i = 1; i < ACTIVE_SHIRES; i *= 2)
    ++level;
  for (size_t i = 0; i <= level; i++)
    checksum = tensor_reduce_uint32(checksum, TENSOR_REDUCE_OP_IADD, i, 0x3);

  // Convert Checksum to ASCII and dump via UART1
  if (minionId == 0) {
    char chs_str[] = "CheckSum: 0xXXXXXXXX\n";
    uint32_to_ascii_hex(&chs_str[12], checksum);
    syscall(SYSCALL_LOG_WRITE, (uint64_t)chs_str, sizeof(chs_str), 0);
  }
}

// Node to force flushing the L3 at the end of the computation
inline void fwdLibFlushL3(uint32_t numShires) {
  uint32_t minion = get_minion_id() & 0x1F;
  // The T0 of minion N of shire 0 flushes the L3 of shire N
  if ((get_shire_id() == 0) && (get_thread_id() == 0) && (minion < numShires)) {
      syscall(SYSCALL_FLUSH_L3, minion, 0, 0);
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _CHECKSUM_H_
