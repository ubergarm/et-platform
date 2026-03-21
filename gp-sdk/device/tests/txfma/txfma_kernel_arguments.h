#ifndef TXFMA_KERNEL_ARGUMENTS_H
#define TXFMA_KERNEL_ARGUMENTS_H

/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */


struct Matrix {
  uint32_t rows;
  uint32_t cols;
  float * data;
} __attribute__ ((packed));

struct KernelArguments {
  Matrix A;
  Matrix B;
  Matrix C;
} __attribute__ ((packed));

#endif

