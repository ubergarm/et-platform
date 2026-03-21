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
int entryPoint_0(KernelArguments* vectors);
void busyWait(short unsigned int n);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

#define CLK_600MHZ_CYCLES_TO_MILISEC 1.6667E-06


int entryPoint_0([[ maybe_unused ]] KernelArguments* vectors) {
  //this test assumes ~4096 bytes per hart trace buffer.
    SCOPED_USER_PROFILE_EVENT("entryPoint");
    {
      SCOPED_USER_PROFILE_EVENT("firstLevel");
      busyWait(1);
      {
        SCOPED_USER_PROFILE_EVENT("secondLevel");
        busyWait(1);
        {
          SCOPED_USER_PROFILE_EVENT("thirdLevel");
          busyWait(1);
        }
      }
    }

  return 0;
}

void busyWait(short unsigned int n) { 
  int64_t start = et_get_timestamp();
  int64_t elapsed = et_get_delta_timestamp(start);
  while (elapsed < 0) {
    elapsed = et_get_delta_timestamp(start);
  }

  SCOPED_USER_PROFILE_EVENT(n);

  while ((elapsed * static_cast<long double>(CLK_600MHZ_CYCLES_TO_MILISEC)) < n) {
    elapsed = et_get_delta_timestamp(start);
    while (elapsed < 0) {
      elapsed = et_get_delta_timestamp(start);
    }
  }
}
