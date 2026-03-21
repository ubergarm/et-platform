/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef USER_DEFINED_STACK_KERNEL_ARGUMENTS_H
#define USER_DEFINED_STACK_KERNEL_ARGUMENTS_H

struct KernelArguments {
  uint64_t stackSize;
} __attribute__((packed));

#endif
