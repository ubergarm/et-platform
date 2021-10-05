/*-------------------------------------------------------------------------
 * Copyright (C) 2019, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
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

constexpr unsigned max_tensor_dimensions = 6; // TODO: deprecate it
constexpr unsigned maxTensorDimensions = 6;

using dim_array_t = std::array<dim_t, maxTensorDimensions>;   // TODO: deprecate
using sdim_array_t = std::array<sdim_t, maxTensorDimensions>; // TODO: deprecate
using dimArray_t = std::array<dim_t, maxTensorDimensions>;
using sdimArray_t = std::array<sdim_t, maxTensorDimensions>;

// An enum representing the type used by the elements of a tensor. The types of Handles for these tensors should match
// the element kind
enum ElemKind {
  // 32-bit float type (float)
  FloatTy,
  // 16-bit float type (half, fp16)
  Float16Ty,
  // 16-bit float type (bfloat16)
  BFloat16Ty,
  // 8-bit quantized type (int8_t)
  Int8QTy,
  // unsigned 8-bit quantized type (uint8_t)
  UInt8QTy,
  // 16-bit quantized type (int16_t)
  Int16QTy,
  // 32-bit quantized type (int32_t)
  Int32QTy,
  // 32-bit index type (int32_t)
  Int32ITy,
  // 64-bit index type (int64_t)
  Int64ITy,
  // 8-bit quantized type with fused scale/offset (uint8_t)
  UInt8FusedQTy,
  // 8-bit quantized type with fused FP16 scale/offset (uint8_t)
  UInt8FusedFP16QTy,
  // 4-bit quantized type with fused FP16 scale/offset (uint8_t, each byte
  // represents 2 4-bit quantized data)
  UInt4FusedFP16QTy,
  // 4-bit quantized type with fused FP32 scale/offset (uint8_t, each byte
  // represents 2 4-bit quantized data)
  UInt4FusedQTy,
  // Bool type (bool)
  BoolTy,
};

// Enum with list of known members
#define SCALAR_MB_DEF(NAME, TYPE, GETTER) mb##NAME,
#define VECTOR_MB_DEF(NAME, TYPE, GETTER) mb##NAME,
enum class instrMembers {
  mbInvalid = 0,
#include "LibApiMembers.def"
  mbMaxMembers
};

// Type and name maps
template <dnn_lib::instrMembers mb> struct memberMap;

// clang-format off
#define SCALAR_MB_DEF(NAME, TYPE, GETTER)                                                                              \
  template <> struct memberMap<dnn_lib::instrMembers::mb##NAME> {                                                                    \
    using type = TYPE;                                                                                                 \
    static const std::string name() {                                                                                  \
      return #NAME;                                                                                                    \
    }                                                                                                                  \
  };

#define VECTOR_MB_DEF(NAME, TYPE, GETTER)                                                                              \
  template <> struct memberMap<dnn_lib::instrMembers::mb##NAME> {                                                                    \
    using type = std::vector<TYPE>;                                                                                    \
    static const std::string name() {                                                                                  \
      return #NAME;                                                                                                    \
    }                                                                                                                  \
  };
// clang-format on

#include "LibApiMembers.def"

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
};

// Local
class LibTensor; // TODO: eventually deprecate

static constexpr size_t maxImplVersions = 4;
static constexpr size_t maxInstrConfigStrLen = 256;
static constexpr size_t maxNrOperands = 12;

// Instruction properties struct
struct instrConfig {
  using operandStateArray = std::array<operandState, maxNrOperands>;
  using implStateArray = std::array<operandStateArray, maxImplVersions + 1>;
  using sel_fnc_t = size_t (*)(std::vector<LibTensor*>&, std::vector<LibTensor*>&);

  char name[maxInstrConfigStrLen];
  size_t nrOutputTensors; // number of output and in/out tensor operands
  size_t nrInputTensors;  // number of input tensor operands
  std::array<instrMembers, (long unsigned int)instrMembers::mbMaxMembers> members;
  uint64_t templateMask;
  std::array<char[maxInstrConfigStrLen], maxImplVersions> versions;
  sel_fnc_t implSel;

  implStateArray stateL1;
  implStateArray stateL2;
  implStateArray stateCB;
  std::array<uint64_t, maxImplVersions + 1> evictAvailableMask;

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
};

/*@brief returns is \p elk is a quantized ElemKind.
 */
inline constexpr bool isQuantizedElemKind(dnn_lib::ElemKind elk) {
  if (elk == dnn_lib::ElemKind::Int8QTy || elk == dnn_lib::ElemKind::UInt8QTy || elk == dnn_lib::ElemKind::Int16QTy ||
      elk == dnn_lib::ElemKind::Int32QTy || elk == dnn_lib::ElemKind::UInt8FusedQTy ||
      elk == dnn_lib::ElemKind::UInt8FusedFP16QTy || elk == dnn_lib::ElemKind::UInt4FusedFP16QTy) {
    return true;
  } else {
    return false;
  }
}

/*@brief returns whether \p elk is an "index" ElemKind.
 */
inline constexpr bool isIndexElemKind(dnn_lib::ElemKind elk) {
  return elk == dnn_lib::ElemKind::Int32ITy or elk == dnn_lib::ElemKind::Int64ITy;
}

/*@brief returns wheter \p elk is a fused quantized ElemKind.
 */
inline constexpr bool isFusedQuantizedElemKind(ElemKind elk) {
  if (elk == dnn_lib::ElemKind::UInt8FusedQTy || elk == dnn_lib::ElemKind::UInt8FusedFP16QTy ||
      elk == dnn_lib::ElemKind::UInt4FusedFP16QTy) {
    return true;
  } else {
    return false;
  }
}
} // namespace dnn_lib

#endif // LIB_TYPES_H
