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
  auto start = et_get_timestamp();
  auto elapsed = et_get_delta_timestamp(start);
  SCOPED_USER_PROFILE_EVENT(n);

  while ((elapsed * static_cast<long double>(CLK_600MHZ_CYCLES_TO_MILISEC)) < n) {
    elapsed = et_get_delta_timestamp(start);
  }
}