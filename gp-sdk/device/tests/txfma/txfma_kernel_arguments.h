#ifndef TXFMA_KERNEL_ARGUMENTS_H
#define TXFMA_KERNEL_ARGUMENTS_H

/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include "environment.h"

struct Matrix {
  uint32_t rows;
  uint32_t cols;
  float * data;
} __attribute__ ((packed));

struct KernelArguments : Arguments {
  Matrix A;
  Matrix B;
  Matrix C;
} __attribute__ ((packed));

#endif

