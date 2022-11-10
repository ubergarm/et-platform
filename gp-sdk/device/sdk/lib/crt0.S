#include <etsoc/isa/syscall.h>

.section .text.startup, "ax", @progbits
.global _start

_start:
.option push
.option norelax
.option pop

  call  deviceGpSdkEntry

  li    a2, KERNEL_RETURN_SUCCESS
  mv    a1, a0
  li    a0, SYSCALL_RETURN_FROM_KERNEL
  ecall
