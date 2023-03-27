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

