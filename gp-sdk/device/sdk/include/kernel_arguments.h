// clang-format off

#ifndef KERNEL_ARGUMENTS_H
#define KERNEL_ARGUMENTS_H

// TODO: make a common header for host and device structures.
struct TensorDesc {
    uint64_t nDims;
    uint64_t dims[6] = {0};
    uint64_t strides[6] = {0};
    void * deviceAddress;
  } __attribute__((packed));
  
struct kernelArguments{
    uint32_t nTensors;
    TensorDesc tensors[4];
    uint32_t operation; 
  } __attribute__((packed));

enum { DENOISE = 1 << 1, CONDITION = 1 << 2, QUANTIZE8 = 1 << 3, SKIP = 1 << 31 };

#endif

