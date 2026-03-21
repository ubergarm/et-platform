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
int entryPoint_1(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, entryPoint_1);

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  if (get_relative_thread_id() < 32) {
    et_printf("%s,%d HELLO WORLD from minion %d hart %d !!!!\n",__func__,__LINE__, get_minion_id(), get_hart_id());
  }
  return 0;
}


int entryPoint_1([[maybe_unused]] KernelArguments* args) {
  if (get_relative_thread_id() < 32) {
    et_printf("%s,%d HELLO WORLD fom minion %d hart %d!!!!\n",__func__,__LINE__, get_minion_id(), get_hart_id());
  }
  return 0;
}

