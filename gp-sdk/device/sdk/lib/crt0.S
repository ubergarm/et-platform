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


#include <etsoc/isa/syscall.h>

.section .text.init, "ax", @progbits
.global _start

_start:
 #initialize global pointer
.option push
.option norelax
 la gp, __global_pointer$
.option pop

  call  deviceGpSdkEntry

  li    a2, KERNEL_RETURN_SUCCESS
  mv    a1, a0
  li    a0, SYSCALL_RETURN_FROM_KERNEL
  ecall
