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

#include <array>
#include <stdio.h>
#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include "entryPoint.h"

constexpr size_t size = 256ULL;
class KernelArguments;
int entryPoint_0(KernelArguments* args);
int entryPoint_1(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, entryPoint_1);
static uint8_t uninitializedData[size];
static uint8_t initToZeroData[size] = {0};

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  bool errorFound = 0;
  if (get_minion_id()==0) {
    for(size_t i = 0; i < size; i++) {
      if (uninitializedData[i] != 0 || initToZeroData[i] != 0) {
        errorFound = 1;
        break;
      }
    }

    et_assert(!errorFound && "Error: .bss section is not set to 0\n");

    if (!errorFound) et_printf("%s() Results are correct.", __func__);
  }
  
  return 0;
}

int entryPoint_1([[maybe_unused]] KernelArguments* args) {
  bool errorFound = 0;
  if (get_minion_id()==0) {
    for(size_t i = 0; i < size; i++) {
      if (uninitializedData[i] != 0 || initToZeroData[i] != 0) {
        errorFound = 1;
        break;
      }
    }

    et_assert(!errorFound && "Error: .bss section is not set to 0\n");

    if (!errorFound) et_printf("%s() Results are correct.", __func__);
  }

  return 0;
}

