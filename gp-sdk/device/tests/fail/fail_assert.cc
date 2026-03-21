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

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  
  if (get_relative_thread_id()==0) {
    et_printf("fail_assert\n");
    et_printf("%s,%d This test has to Fail due to assert!\n",__func__,__LINE__);
    et_assert(false==true);    
  }

  return 0;
}
