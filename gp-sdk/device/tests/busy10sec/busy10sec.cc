/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>

#include "entryPoint.h"

class KernelArguments;
int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

#define CLK_600MHZ_CYCLES_TO_SEC 1.6667E-09

int entryPoint_0([[maybe_unused]] KernelArguments* args) {

  auto threadId = get_relative_thread_id();
  decltype(threadId) assignedMinions = 32;

  if (threadId >= assignedMinions) {
    return 0;
  }
  auto start = et_get_timestamp();
  auto elapsed = et_get_delta_timestamp(start);

  while ((elapsed * static_cast<long double>(CLK_600MHZ_CYCLES_TO_SEC)) < 10) {
    elapsed = et_get_delta_timestamp(start);
  }

  et_printf("Success\n");

  return 0;
}
