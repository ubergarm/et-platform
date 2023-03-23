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

#include "entryPoint.h"

class KernelArguments;
int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

#define CLK_600MHZ_CYCLES_TO_SEC 1.6667E-09

int entryPoint_0([[maybe_unused]] KernelArguments* args) {

  auto minionId = get_minion_id();
  uint64_t assignedMinions = 32;

  if (minionId >= assignedMinions) {
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
