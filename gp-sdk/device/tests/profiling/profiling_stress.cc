/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>
#include <algorithm>

#include "entryPoint.h"
#include "profiling.h"

class KernelArguments;



int entryPoint(KernelArguments *);

DECLARE_KERNEL_ENTRY_POINTS(entryPoint, entryPoint);

int entryPoint([[ maybe_unused ]] KernelArguments * vectors) {
  // this test assumes ~4096 bytes per hart trace buffer.
  // 200 is enough to stress the circular buffer wrap-around (4096 bytes) few times.
  // goal is getting all the writes
  constexpr size_t kMaxIters = 200;
  for(size_t i = 0; i < kMaxIters; i ++) {
    SCOPED_USER_PROFILE_EVENT(1);
  }
  return 0;
}
