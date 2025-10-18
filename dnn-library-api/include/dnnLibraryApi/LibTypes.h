/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef LIB_TYPES_H
#define LIB_TYPES_H

// STD
#include <array>
#include <stdint.h>
#include <string>
#include <vector>

namespace dnn_lib {

#ifdef DIM_T_32
using dim_t = uint32_t;
using sdim_t = int32_t;
#else
using dim_t = size_t;
using sdim_t = int64_t;
#endif

constexpr unsigned maxTensorDimensions = 6;

using dim_array_t = std::array<dim_t, maxTensorDimensions>;   // TODO: deprecate
using dimArray_t = std::array<dim_t, maxTensorDimensions>;

// An enum representing the type used by the elements of a tensor. The types of Handles for these tensors should match
// the element kind
enum ElemKind {
  FloatTy,           // 32-bit float type (float)
  Float16Ty,         // 16-bit float type (half, fp16)
  BFloat16Ty,        // 16-bit float type (bfloat16)
  Int8QTy,           // 8-bit quantized type (int8_t)
  UInt8QTy,          // unsigned 8-bit quantized type (uint8_t)
  Int16QTy,          // 16-bit quantized type (int16_t)
  Int32QTy,          // 32-bit quantized type (int32_t)
  Int32ITy,          // 32-bit index type (int32_t)
  Int64ITy,          // 64-bit index type (int64_t)
  UInt8FusedQTy,     // 8-bit quantized type with fused scale/offset (uint8_t)
  UInt8FusedFP16QTy, // 8-bit quantized type with fused FP16 scale/offset (uint8_t)
  UInt4FusedFP16QTy, // 4-bit quantized type with fused FP16 scale/offset (uint8_t, each byte represents 2 4-bit
                     // quantized data)
  UInt4FusedQTy, // 4-bit quantized type with fused FP32 scale/offset (uint8_t, each byte represents 2 4-bit quantized
                 // data)
  BoolTy,        // Bool type (bool)
};

enum class InstrMembers;

// Cache operand state after operation
enum class operandState { dirty, clean, untouched };

// Tensor information for the operands
struct Tensor {
  ElemKind elementType;    // Element type of the contents
  dimArray_t sizes;        // Sizes of the dimensions
  dimArray_t strides;      // Strides for each dimension
  dim_t numDims;           // Number of dimensions
  float scale;             // Quantization scale
  int32_t offset;          // Quantization offset
  uint64_t alignOffset;    // Offset within an initial address that might unalign the tensor
  bool untouchablePadding; // If the padding of the tensor is untouchable
  bool hasSingleValue;     // If all the values of the tensor are the same
  float singleValue;       // Repeated value if hasSingleValue is set
  bool isCounter;          // If the tensor is a counter of increasing numbers with same stride
  int64_t counterOffset;   // Starting value of the counter
  int64_t counterStride;   // Stride between values of the counter
};

// Instruction properties struct
struct instrConfig {
  using implStateVector = std::vector<std::vector<operandState>>;

  std::string name;
  size_t nrOutputTensors; // number of output and in/out tensor operands
  size_t nrInputTensors;  // number of input tensor operands
  std::vector<InstrMembers> members;
  uint64_t templateMask;
  std::vector<std::string> versions;

  implStateVector stateL1;
  implStateVector stateL2;
  implStateVector stateCB;
  std::vector<uint64_t> evictAvailableMask;
  std::vector<uint64_t> dstGlobalStore;

  // functions to retrieve operand information
  operandState getOperandStateL1(size_t implIdx, size_t operand);
  operandState getOperandStateL2(size_t implIdx, size_t operand);
  operandState getOperandStateCB(size_t implIdx, size_t operand);
  bool getOperandAutoEvict(size_t implIdx, size_t operand);

  // and same as before, but index is either input or output
  operandState getSrcStateL1(size_t implIdx, size_t idx);
  operandState getSrcStateL2(size_t implIdx, size_t idx);
  operandState getSrcStateCB(size_t implIdx, size_t idx);
  bool getSrcAutoEvict(size_t implIdx, size_t idx);

  operandState getDstStateL1(size_t implIdx, size_t idx);
  operandState getDstStateL2(size_t implIdx, size_t idx);
  operandState getDstStateCB(size_t implIdx, size_t idx);
  bool getDstAutoEvict(size_t implIdx, size_t idx);
  bool getDstGlobalStore(size_t implIdx, size_t idx);
  uint64_t getDstGlobalStore(size_t implIdx);
};

} // namespace dnn_lib

#endif // LIB_TYPES_H
