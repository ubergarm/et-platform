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

int stepvar = 0;

int entryPoint_0([[maybe_unused]] KernelArguments* args) {

  auto hartId = get_hart_id();
  
  if (hartId == 0 || hartId == 2) {
    et_printf("%s,%d HELLO WORLD before COHERENCE  hart:%d -- stepvar=%d !!!!\n",__func__,__LINE__,hartId, stepvar);
    stepvar++;    
  }

  return 0;
}
