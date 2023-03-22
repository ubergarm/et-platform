/*-------------------------------------------------------------------------
 * Copyright (C) 2023, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>
#include <algorithm>

#include "entryPoint.h"
#include "profiling.h"

class KernelArguments;



int entryPoint(KernelArguments *);

DECLARE_KERNEL_ENTRY_POINTS(1, entryPoint, nullptr);


int entryPoint([[ maybe_unused ]] KernelArguments * vectors) {
  //this test assumes ~4096 bytes per hart trace buffer.
  //1000 is enough to stress the circular buffer wrap-around (4096 bytes) few times.
  //goal is getting all the writes
  constexpr size_t kMaxIters = 1000;
  for(size_t i = 0; i < kMaxIters; i ++) {
    SCOPED_USER_PROFILE_EVENT(1);
  }
  return 0;
}
