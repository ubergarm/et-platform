/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */


#include "entryPoint.h"
class kernelArguments;
#include <etsoc/common/utils.h>

int entryPoint_0(kernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, entryPoint_0);

extern "C" int uberKernel_RAWKERNEL_entry_point(kernelArguments * layer_dyn_info);

int entryPoint_0(kernelArguments* args) {
  return uberKernel_RAWKERNEL_entry_point((kernelArguments *)args);;
}

