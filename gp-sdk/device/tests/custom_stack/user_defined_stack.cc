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
#include <stddef.h>
#include <stdint.h>

#include "entryPoint.h"
#include "user_defined_stack_kernel_arguments.h"

class KernelArguments;

int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  if (args->stackSize > 0) {
    if ((get_relative_thread_id() == 0) || (get_relative_thread_id() == 2)) {
      unsigned long int* ptr;
      asm volatile("mv %[ptr], sp  \n" : [ ptr ] "=r"(ptr));

      et_printf("Stack size received: 0x%lx Stack ptr:0x%lx\r\n", args->stackSize, (unsigned long int)ptr);

      /* Allocate on stack and do a dummy copy
      64-bytes of stack are consumed for this kernel, hence allocate rest */
      char arr[args->stackSize - 64];
      et_memset(arr, 'A', args->stackSize - 64);
    }
  } else {
    et_printf("Stack size not defined\r\n");
  }
  return 0;
}
