#ifndef BARRIER_KERNEL_ARGUMENTS_H
#define BARRIER_KERNEL_ARGUMENTS_H

/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */


struct KernelArguments {
  uint64_t * data;
  uint64_t * accumData;
} __attribute__ ((packed));

#endif

