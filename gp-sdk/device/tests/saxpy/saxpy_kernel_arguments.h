#ifndef SAXPY_KERNEL_ARGUMENTS_H
#define SAXPY_KERNEL_ARGUMENTS_H

/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */


struct KernelArguments {
  uint64_t numElements;
  float * x;
  float * y;
  float a;
} __attribute__ ((packed));

#endif

