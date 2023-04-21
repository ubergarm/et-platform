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


#include "entryPoint.h"
class kernelArguments;
#include <etsoc/common/utils.h>

int entryPoint_0(kernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, entryPoint_0);

extern "C" int uberKernel_RAWKERNEL_entry_point(kernelArguments * layer_dyn_info);

int entryPoint_0(kernelArguments* args) {
  return uberKernel_RAWKERNEL_entry_point((kernelArguments *)args);;
}

