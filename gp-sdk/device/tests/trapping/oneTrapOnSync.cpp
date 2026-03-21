
/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
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
  auto minionId = get_relative_thread_id();
  decltype(minionId) assignedMinions = 32;
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
