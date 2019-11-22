/*-------------------------------------------------------------------------
 * Copyright (C) 2018, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include <device_common.h>
#include <syscall.h>

#include "LibNodes.h"
#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"

using namespace dnn_lib;
using namespace std;

#define GEN_INSTANCES(functionName, op, ...)                                   \
  template void functionName<float, op>(__VA_ARGS__);                          \
  template void functionName<float16, op>(__VA_ARGS__);                        \
  template void functionName<int8_t, op>(__VA_ARGS__);                         \
  template void functionName<int64_t, op>(__VA_ARGS__);

#define GEN_OP(functionName, ...)                                              \
  template void functionName<float>(__VA_ARGS__);                              \
  template void functionName<float16>(__VA_ARGS__);                            \
  template void functionName<int8_t>(__VA_ARGS__);                             \
  template void functionName<int64_t>(__VA_ARGS__);                            \
  template void functionName<int32_t>(__VA_ARGS__);                            \
  template void functionName<int16_t>(__VA_ARGS__);

#define GEN_1TYPEFP(functionName, ...)                                         \
  template void functionName<float>(__VA_ARGS__);                              \
  template void functionName<float16>(__VA_ARGS__);

#define GEN_3TYPE(functionName, op, ...)                                       \
  template void functionName<float, float, float, op>(__VA_ARGS__);            \
  template void functionName<float16, float16, float16, op>(__VA_ARGS__);      \
  template void functionName<int8_t, int8_t, int8_t, op>(__VA_ARGS__);         \
  template void functionName<uint8_t, int8_t, int8_t, op>(__VA_ARGS__);        \
  template void functionName<int8_t, uint8_t, int8_t, op>(__VA_ARGS__);        \
  template void functionName<int8_t, int8_t, uint8_t, op>(__VA_ARGS__);        \
  template void functionName<uint8_t, uint8_t, int8_t, op>(__VA_ARGS__);       \
  template void functionName<uint8_t, int8_t, uint8_t, op>(__VA_ARGS__);       \
  template void functionName<int8_t, uint8_t, uint8_t, op>(__VA_ARGS__);       \
  template void functionName<uint8_t, uint8_t, uint8_t, op>(__VA_ARGS__);      \
  template void functionName<int64_t, int64_t, int64_t, op>(__VA_ARGS__);

#define GEN_2TYPE(functionName, op, ...)                                       \
  template void functionName<float, float, op>(__VA_ARGS__);                   \
  template void functionName<float16, float16, op>(__VA_ARGS__);               \
  template void functionName<int8_t, int8_t, op>(__VA_ARGS__);                 \
  template void functionName<uint8_t, int8_t, op>(__VA_ARGS__);                \
  template void functionName<int8_t, uint8_t, op>(__VA_ARGS__);                \
  template void functionName<uint8_t, uint8_t, op>(__VA_ARGS__);               \
  template void functionName<int64_t, int64_t, op>(__VA_ARGS__);

#define GEN_3TYPE_OP(functionName, ...)                                        \
  template void functionName<float, float, float>(__VA_ARGS__);                \
  template void functionName<float16, float16, float16>(__VA_ARGS__);          \
  template void functionName<int8_t, int8_t, int8_t>(__VA_ARGS__);             \
  template void functionName<uint8_t, int8_t, int8_t>(__VA_ARGS__);            \
  template void functionName<int8_t, uint8_t, int8_t>(__VA_ARGS__);            \
  template void functionName<int8_t, int8_t, uint8_t>(__VA_ARGS__);            \
  template void functionName<uint8_t, uint8_t, int8_t>(__VA_ARGS__);           \
  template void functionName<uint8_t, int8_t, uint8_t>(__VA_ARGS__);           \
  template void functionName<int8_t, uint8_t, uint8_t>(__VA_ARGS__);           \
  template void functionName<uint8_t, uint8_t, uint8_t>(__VA_ARGS__);          \
  template void functionName<int64_t, int64_t, int64_t>(__VA_ARGS__);

#define GEN_2TYPE_OP(functionName, ...)                                        \
  template void functionName<float, float>(__VA_ARGS__);                       \
  template void functionName<float16, float16>(__VA_ARGS__);                   \
  template void functionName<int8_t, int8_t>(__VA_ARGS__);                     \
  template void functionName<uint8_t, int8_t>(__VA_ARGS__);                    \
  template void functionName<int8_t, uint8_t>(__VA_ARGS__);                    \
  template void functionName<uint8_t, uint8_t>(__VA_ARGS__);                   \
  template void functionName<int64_t, int64_t>(__VA_ARGS__);

#define GEN_INTONLY_OP(functionName, ...)                                      \
  template void functionName<int64_t>(__VA_ARGS__);                            \
  template void functionName<int32_t>(__VA_ARGS__);

#define GEN_QUANT(functionName, ...)                                           \
  template void functionName<int8_t>(__VA_ARGS__);                             \
  template void functionName<int16_t>(__VA_ARGS__);                            \
  template void functionName<int32_t>(__VA_ARGS__);

#define GEN_OP_INDEX(functionName, ...)                                        \
  template void functionName<float, int64_t>(__VA_ARGS__);                     \
  template void functionName<float16, int64_t>(__VA_ARGS__);                   \
  template void functionName<int8_t, int64_t>(__VA_ARGS__);                    \
  template void functionName<int64_t, int64_t>(__VA_ARGS__);                   \
  template void functionName<int32_t, int64_t>(__VA_ARGS__);                   \
  template void functionName<float, int32_t>(__VA_ARGS__);                     \
  template void functionName<float16, int32_t>(__VA_ARGS__);                   \
  template void functionName<int8_t, int32_t>(__VA_ARGS__);                    \
  template void functionName<int64_t, int32_t>(__VA_ARGS__);                   \
  template void functionName<int32_t, int32_t>(__VA_ARGS__);

#define GEN_CONVERT(functionName, ...)                                         \
  template void functionName<float, int64_t>(__VA_ARGS__);                     \
  template void functionName<float, float16>(__VA_ARGS__);                     \
  template void functionName<float, float>(__VA_ARGS__);                       \
  template void functionName<float16, float>(__VA_ARGS__);                     \
  template void functionName<float16, float16>(__VA_ARGS__);                   \
  template void functionName<int64_t, float>(__VA_ARGS__);                     \
  template void functionName<int64_t, int64_t>(__VA_ARGS__);

#define GEN_FRQSLWS_V(functionName, ...)                                       \
  template void functionName<true,  true,  true>(__VA_ARGS__);                  \
  template void functionName<true,  true,  false>(__VA_ARGS__);                 \
  template void functionName<true,  false, true>(__VA_ARGS__);                  \
  template void functionName<false, true,  false>(__VA_ARGS__);                 \
  template void functionName<false, false, true>(__VA_ARGS__);

namespace dnn_lib {

#include "AutoGenInstan.def"

}

// TODO: Remove this and properly compile them
#include "AvgPoolInst.cpp"
#include "BatchedAddInst.cpp"
#include "BatchedReduceAddInst.cpp"
#include "BatchOneHotInst.cpp"
#include "Checksum.cpp"
#include "ConvertToInst.cpp"
#include "Convolution3DInst.cpp"
#include "ConvolutionInst.cpp"
#include "CopyInst.cpp"
#include "CopyInstTensorized.cpp"
#include "CrossEntropyLossInst.cpp"
#include "DequantizeInst.cpp"
#include "ElementBoolInst.cpp"
#include "ElementExpInst.cpp"
#include "ElementInst.cpp"
#include "ElementIsNaNInst.cpp"
#include "ElementSelectInst.cpp"
#include "ElementSingleInst.cpp"
#include "ETSOCFullyConnectedInst.cpp"
#include "ETSOCMaxSplatInst.cpp"
#include "ExtractTensorInst.cpp"
#include "FusedRowwiseQuantizedSparseLengthsSum.cpp"
#include "FusedRowwiseQuantizedSparseLengthsWeightedSum.cpp"
#include "GatherInst.cpp"
#include "GatherRangesInst.cpp"
#include "InsertTensorInst.cpp"
#include "IntLookupTable.cpp"
#include "LengthsSumInst.cpp"
#include "LengthsToRangesInst.cpp"
#include "LocalResponseNormalizationInst.cpp"
#include "MatMulInst.cpp"
#include "MatMulInstTransposed.cpp"
#include "MaxPoolInst.cpp"
#include "ModuloInst.cpp"
#include "QuantizeInst.cpp"
#include "RescaleQuantizedInst.cpp"
#include "RowwiseQuantizedFullyConnected.cpp"
#include "RowwiseQuantizedSparseLengthsWeightedSum.cpp"
#include "ScatterAssignInst.cpp"
#include "SigmoidInst.cpp"
#include "SoftMaxInst1.cpp"
#include "SoftMaxInst2.cpp"
#include "SoftMaxInst.cpp"
#include "SparseLengthsWeightedSumInst.cpp"
#include "SparseToDenseInst.cpp"
#include "SparseToDenseMaskInst.cpp"
#include "SplatInst.cpp"
#include "TanhInst.cpp"
#include "TensorViewInst.cpp"
#include "TopKInst.cpp"
#include "TransposeInst.cpp"
