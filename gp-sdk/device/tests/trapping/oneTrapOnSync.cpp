
/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
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

#include <etsoc/isa/utils.h>

#include "entryPoint.h"

#include "sync.h"

class KernelArguments;
int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  // Set a barrier all assinged minions and one
  auto minionId = get_minion_id();
  uint64_t assignedMinions = 32;
  if (minionId >= assignedMinions) {
    return 0;
  }

  if (minionId == 0) {
    // Generate code exception
    *(volatile uint64_t*)0 = 0xDEADBEEF;
  }

  hart::barrier();

  et_printf("Success\n");

  return 0;
}
