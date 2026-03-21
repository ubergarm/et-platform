#ifndef CODEGEN_MATMUL_ARGUMENTS_H
#define CODEGEN_MATMUL_ARGUMENTS_H

/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

typedef struct {
  void * B;
  void * PH;
  void * tracing_0;
} __attribute__((packed)) kernelArguments;

#endif
