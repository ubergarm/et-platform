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

#ifndef USER_DEFINED_STACK_KERNEL_ARGUMENTS_H
#define USER_DEFINED_STACK_KERNEL_ARGUMENTS_H

struct KernelArguments {
  uint64_t stackSize;
} __attribute__((packed));

#endif
