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

