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
int factorial(int n);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

int entryPoint_0([[ maybe_unused ]] KernelArguments* vectors) {

  //this test assumes ~4096 bytes per hart trace buffer.
    SCOPED_USER_PROFILE_EVENT("entryPoint");
    {
      SCOPED_USER_PROFILE_EVENT("firstLevel");
      factorial(4);
      {
        SCOPED_USER_PROFILE_EVENT("secondLevel");
        factorial(4);
        {
          SCOPED_USER_PROFILE_EVENT("thirdLevel");
          factorial(4);
          {
            SCOPED_USER_PROFILE_EVENT("fourthLevel");
            factorial(4);
          }
        }
      }
    }

  return 0;
}

int factorial(int n) {
  SCOPED_USER_PROFILE_EVENT(n);
    if (n == 0 || n == 1) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}