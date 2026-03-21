/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
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
