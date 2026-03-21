/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */
#ifndef KERNEL_ARGUMENTS_H
#define KERNEL_ARGUMENTS_H


// TODO: make a common header for host and device structures.
struct TensorDesc {
    uint64_t nDims;
    uint64_t dims[6] = {0};
    uint64_t strides[6] = {0};
    uint64_t deviceAddress;
 } __attribute__((packed));

struct KernelArguments { 
    uint32_t nTensors;
    TensorDesc tensors[2];
    uint32_t operation;
  } __attribute__((packed));

  enum class FFTOp { FFT = 1, IFFT = 2, SKIP = 1024};

#endif

