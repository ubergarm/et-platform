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

#ifndef LIBNODES_H_
#define LIBNODES_H_

#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

class Add {};
class Sub {};
class Mul {};
class Div {};
class Max {};
class Min {};
class CmpEQ {};
class CmpLTE {};
class Select {};
class Pow {};
class ElementLog {};

using namespace std;

namespace dnn_lib {

enum PrecisionMode {
  // TODO: Get same enumerate as Jitter
  PM_FP_32 = 0,   // fp32
  PM_FP_16 = 1,   // fp16
  PM_INT_32 = 2,  // quant int32
  PM_INT_8 = 3,   // quant int8
  PM_INT_16 = 4,  // quant int16
  PM_INT_I32 = 5, // idx int32
  PM_INT_I64 = 6, // idx int64
  PM_UINT_8 = 7,  // quant uint8
  PM_BOOL = 8,    // bool
  MAX_PRECISION_MODES
};

#define dispatchLibImplEltWiseSingle(functionName, pm1, op, ...)                                            \
  switch (pm1) {                                                                                           \
  case dnn_lib::PrecisionMode::PM_FP_32:                                                                   \
    functionName<float, op>(__VA_ARGS__);                                                           \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_FP_16:                                                                   \
    functionName<float16, op>(__VA_ARGS__);                                                       \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_INT_8:                                                                   \
    functionName<int8_t, op>(__VA_ARGS__);                                                       \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_UINT_8:                                                                  \
    functionName<uint8_t, op>(__VA_ARGS__);                                                      \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_INT_I64:                                                                 \
    functionName<int64_t, op>(__VA_ARGS__);                                                       \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_BOOL:                                                                    \
    functionName<bool, op>(__VA_ARGS__);                                                             \
    break;                                                                                                 \
  default:                                                                                                 \
    break;                                                                                                 \
  }

#define dispatchLibImplEltWise(functionName, pm1, pm2, op, ...)                                            \
  switch (pm1) {                                                                                           \
  case dnn_lib::PrecisionMode::PM_FP_32:                                                                   \
    functionName<float, float, op>(__VA_ARGS__);                                                           \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_FP_16:                                                                   \
    functionName<float16, float16, op>(__VA_ARGS__);                                                       \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_INT_8:                                                                   \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_8) {                                                         \
      functionName<int8_t, int8_t, op>(__VA_ARGS__);                                                       \
    } else {                                                                                               \
      functionName<int8_t, uint8_t, op>(__VA_ARGS__);                                                      \
    }                                                                                                      \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_UINT_8:                                                                  \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_8) {                                                         \
      functionName<uint8_t, int8_t, op>(__VA_ARGS__);                                                      \
    } else {                                                                                               \
      functionName<uint8_t, uint8_t, op>(__VA_ARGS__);                                                     \
    }                                                                                                      \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_INT_I64:                                                                 \
    functionName<int64_t, int64_t, op>(__VA_ARGS__);                                                       \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_BOOL:                                                                    \
    functionName<bool, bool, op>(__VA_ARGS__);                                                             \
    break;                                                                                                 \
  default:                                                                                                 \
    break;                                                                                                 \
  }

#define dispatchLibImplEltWiseParal(functionName, pm1, pm2, pm3, op, ...)                                  \
  switch (pm1) {                                                                                           \
  case dnn_lib::PrecisionMode::PM_FP_32:                                                                   \
    functionName<float, float, float, op>(__VA_ARGS__);                                                    \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_FP_16:                                                                   \
    functionName<float16, float16, float16, op>(__VA_ARGS__);                                     \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_INT_8:                                                                   \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_8 && pm3 == dnn_lib::PrecisionMode::PM_INT_8) {              \
      functionName<int8_t, int8_t, int8_t, op>(__VA_ARGS__);                                               \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_INT_8 && pm3 == dnn_lib::PrecisionMode::PM_UINT_8) {      \
      functionName<int8_t, int8_t, uint8_t, op>(__VA_ARGS__);                                              \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_UINT_8 && pm3 == dnn_lib::PrecisionMode::PM_INT_8) {      \
      functionName<int8_t, uint8_t, int8_t, op>(__VA_ARGS__);                                              \
    } else {                                                                                               \
      functionName<int8_t, uint8_t, uint8_t, op>(__VA_ARGS__);                                             \
    }                                                                                                      \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_UINT_8:                                                                  \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_8 && pm3 == dnn_lib::PrecisionMode::PM_INT_8) {              \
      functionName<uint8_t, int8_t, int8_t, op>(__VA_ARGS__);                                              \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_INT_8 && pm3 == dnn_lib::PrecisionMode::PM_UINT_8) {      \
      functionName<uint8_t, int8_t, uint8_t, op>(__VA_ARGS__);                                             \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_UINT_8 && pm3 == dnn_lib::PrecisionMode::PM_INT_8) {      \
      functionName<uint8_t, uint8_t, int8_t, op>(__VA_ARGS__);                                             \
    } else {                                                                                               \
      functionName<uint8_t, uint8_t, uint8_t, op>(__VA_ARGS__);                                            \
    }                                                                                                      \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_INT_I64:                                                                 \
    functionName<int64_t, int64_t, int64_t, op>(__VA_ARGS__);                                                       \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_BOOL:                                                                    \
    functionName<bool, bool, bool, op>(__VA_ARGS__);                                                             \
    break;                                                                                                 \
  default:                                                                                                 \
    break;                                                                                                 \
  }

#define dispatchLibImpl2Types(functionName, pm1, pm2, ...)                                                 \
  switch (pm1) {                                                                                           \
  case dnn_lib::PrecisionMode::PM_FP_32:                                                                   \
    functionName<float, float>(__VA_ARGS__);                                                               \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_FP_16:                                                                   \
    functionName<float16, float16>(__VA_ARGS__);                                                                    \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_INT_8:                                                                   \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_8) {                                                         \
      functionName<int8_t, int8_t>(__VA_ARGS__);                                                           \
    } else {                                                                                               \
      functionName<int8_t, uint8_t>(__VA_ARGS__);                                                          \
    }                                                                                                      \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_UINT_8:                                                                  \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_8) {                                                         \
      functionName<uint8_t, int8_t>(__VA_ARGS__);                                                          \
    } else {                                                                                               \
      functionName<uint8_t, uint8_t>(__VA_ARGS__);                                                         \
    }                                                                                                      \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_INT_I64:                                                                 \
    functionName<int64_t, int64_t>(__VA_ARGS__);                                                           \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_BOOL:                                                                    \
    functionName<bool, bool>(__VA_ARGS__);                                                                 \
    break;                                                                                                 \
  default:                                                                                                 \
    break;                                                                                                 \
  }

#define dispatchLibImpl3Types(functionName, pm1, pm2, pm3, ...)                                            \
  switch (pm1) {                                                                                           \
  case dnn_lib::PrecisionMode::PM_FP_32:                                                                   \
    functionName<float, float, float>(__VA_ARGS__);                                                        \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_FP_16:                                                                   \
    functionName<float16, float16, float16>(__VA_ARGS__);                                         \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_INT_8:                                                                   \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_8 && pm3 == dnn_lib::PrecisionMode::PM_INT_8) {              \
      functionName<int8_t, int8_t, int8_t>(__VA_ARGS__);                                                   \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_INT_8 && pm3 == dnn_lib::PrecisionMode::PM_UINT_8) {      \
      functionName<int8_t, int8_t, uint8_t>(__VA_ARGS__);                                                  \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_UINT_8 && pm3 == dnn_lib::PrecisionMode::PM_INT_8) {      \
      functionName<int8_t, uint8_t, int8_t>(__VA_ARGS__);                                                  \
    } else {                                                                                               \
      functionName<int8_t, uint8_t, uint8_t>(__VA_ARGS__);                                                 \
    }                                                                                                      \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_UINT_8:                                                                  \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_8 && pm3 == dnn_lib::PrecisionMode::PM_INT_8) {              \
      functionName<uint8_t, int8_t, int8_t>(__VA_ARGS__);                                                  \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_INT_8 && pm3 == dnn_lib::PrecisionMode::PM_UINT_8) {      \
      functionName<uint8_t, int8_t, uint8_t>(__VA_ARGS__);                                                 \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_UINT_8 && pm3 == dnn_lib::PrecisionMode::PM_INT_8) {      \
      functionName<uint8_t, uint8_t, int8_t>(__VA_ARGS__);                                                 \
    } else {                                                                                               \
      functionName<uint8_t, uint8_t, uint8_t>(__VA_ARGS__);                                                \
    }                                                                                                      \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_INT_I64:                                                                 \
    functionName<int64_t, int64_t, int64_t>(__VA_ARGS__);                                                  \
    break;                                                                                                 \
  case dnn_lib::PrecisionMode::PM_BOOL:                                                                    \
    functionName<bool, bool, bool>(__VA_ARGS__);                                                           \
    break;                                                                                                 \
  default:                                                                                                 \
    break;                                                                                                 \
  }

#define dispatchLibImpl(functionName, pm, ...)                                 \
  switch (pm) {                                                                \
  case dnn_lib::PrecisionMode::PM_FP_32:                                       \
    functionName<float>(__VA_ARGS__);                                          \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_FP_16:                                       \
    functionName<float16>(__VA_ARGS__);                                        \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_8:                                       \
    functionName<int8_t>(__VA_ARGS__);                                         \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_16:                                      \
    functionName<int16_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_I64:                                     \
    functionName<int64_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_I32:                                     \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_BOOL:                                        \
    functionName<bool>(__VA_ARGS__);                                           \
    break;                                                                     \
  default:                                                                     \
    break;                                                                     \
  }

#define dispatchLibQuantizedTyImpl(functionName, pm, ...)                      \
  switch (pm) {                                                                \
  case dnn_lib::PrecisionMode::PM_INT_8:                                       \
    functionName<int8_t>(__VA_ARGS__);                                         \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_16:                                      \
    functionName<int16_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_32:                                      \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    break;                                                                     \
  }

#define dispatchLibWithIndexImpl(functionName, pm1, pm2, ...)                  \
  switch (pm1) {                                                               \
  case dnn_lib::PrecisionMode::PM_FP_32:                                       \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_I64) {                           \
      functionName<float, int64_t>(__VA_ARGS__);                               \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_INT_I32) {                    \
      functionName<float, int32_t>(__VA_ARGS__);                               \
    }                                                                          \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_FP_16:                                       \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_I64) {                           \
      functionName<float16, int64_t>(__VA_ARGS__);                             \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_INT_I32) {                    \
      functionName<float16, int32_t>(__VA_ARGS__);                             \
    }                                                                          \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_8:                                       \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_I64) {                           \
      functionName<int8_t, int64_t>(__VA_ARGS__);                              \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_INT_I32) {                    \
      functionName<int8_t, int32_t>(__VA_ARGS__);                              \
    }                                                                          \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_I64:                                     \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_I64) {                           \
      functionName<int64_t, int64_t>(__VA_ARGS__);                             \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_INT_I32) {                    \
      functionName<int64_t, int32_t>(__VA_ARGS__);                             \
    }                                                                          \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_I32:                                     \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_I64) {                           \
      functionName<int32_t, int64_t>(__VA_ARGS__);                             \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_INT_I32) {                    \
      functionName<int32_t, int32_t>(__VA_ARGS__);                             \
    }                                                                          \
    break;                                                                     \
  default:                                                                     \
    break;                                                                     \
  }

#define dispatchLibConvertImpl(functionName, pm1, pm2, ...)                    \
  switch (pm1) {                                                               \
  case dnn_lib::PrecisionMode::PM_FP_32:                                       \
    if (pm2 == dnn_lib::PrecisionMode::PM_INT_I64) {                           \
      functionName<float, int64_t>(__VA_ARGS__);                               \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_FP_16) {                      \
      functionName<float, float16>(__VA_ARGS__);                               \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_FP_32) {                      \
      functionName<float, float>(__VA_ARGS__);                                 \
    }                                                                          \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_FP_16:                                       \
    if (pm2 == dnn_lib::PrecisionMode::PM_FP_32) {                             \
      functionName<float16, float>(__VA_ARGS__);                               \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_FP_16) {                      \
      functionName<float16, float16>(__VA_ARGS__);                             \
    }                                                                          \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_I64:                                     \
    if (pm2 == dnn_lib::PrecisionMode::PM_FP_32) {                             \
      functionName<int64_t, float>(__VA_ARGS__);                               \
    } else if (pm2 == dnn_lib::PrecisionMode::PM_INT_I64) {                    \
      functionName<int64_t, int64_t>(__VA_ARGS__);                             \
    }                                                                          \
    break;                                                                     \
  default:                                                                     \
    break;                                                                     \
  }

#define dispatchLibIntTyImpl(functionName, pm, ...)                            \
  switch (pm) {                                                                \
  case dnn_lib::PrecisionMode::PM_INT_I64:                                     \
    functionName<int64_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  case dnn_lib::PrecisionMode::PM_INT_32:                                      \
    functionName<int32_t>(__VA_ARGS__);                                        \
    break;                                                                     \
  default:                                                                     \
    break;                                                                     \
  }

#define GEN_INSTANCES(functionName, op, ...)                                   \
  template <typename srcType, typename opType> void functionName(__VA_ARGS__);

#define GEN_OP(functionName, ...)                                              \
  template <typename srcType> void functionName(__VA_ARGS__);

#define GEN_3TYPE(functionName, op, ...)                                   \
  template <typename src1Type, typename src2Type, typename dstType, typename opType> void functionName(__VA_ARGS__); 

#define GEN_2TYPE(functionName, op, ...)                                   \
  template <typename src1Type, typename dstType, typename opType> void functionName(__VA_ARGS__);

#define GEN_3TYPE_OP(functionName, ...)                                              \
  template <typename src1Type, typename src2Type, typename dstType> void functionName(__VA_ARGS__);                  

#define GEN_2TYPE_OP(functionName, ...)                                              \
  template <typename src1Type, typename dstType> void functionName(__VA_ARGS__);

#define GEN_INTONLY_OP(functionName, ...)                                      \
  template <typename srcType> void functionName(__VA_ARGS__);

#define GEN_QUANT(functionName, ...)                                           \
  template <typename srcType> void functionName(__VA_ARGS__);

#define GEN_OP_INDEX(functionName, ...)                                        \
  template <typename srcType, typename indexType>                              \
  void functionName(__VA_ARGS__);

#define GEN_CONVERT(functionName, ...)                                         \
  template <typename srcType, typename dstType> void functionName(__VA_ARGS__);

#define GEN_INT8_FUN(functionName, ...) void functionNameInt8(__VA_ARGS__);

#include "AutoGenInstan.def"

template <class SrcTy, class DestTy> DestTy clip(SrcTy in);
template <class DestTy = int8_t>
inline DestTy quantize(float input, float scale, int32_t offset);
template <class eTy>
inline float dequantizeWithFloatOffset(eTy input, float scale, float offset);
template <class eTy = int8_t>
float dequantize(eTy input, float scale, int32_t offset);
int8_t quantizeValInt8(float val, float scale, int32_t offset);

void fwdLibBatchedAddInsti8i32(void *pdst, void *pdstDims, void *pdstPitches,
                               void *pbatch, void *pbatchDims,
                               void *pbatchPitches, unsigned int pbatchDimNum,
                               void *pslice, void *pslicePitches, float *scale,
                               int32_t *offset);
void fwdLibBatchedReduceAddInstInt8(void *pdst, void *pdstDims,
                                    void *pdstPitches, void *pbatch,
                                    void *pbatchDims, void *pbatchPitches,
                                    unsigned int pbatchDimNum,
                                    unsigned int axis, float *scale,
                                    int32_t *offset);
void fwdLibBatchedReduceAddInstInt8Threaded(void *pdst, void *pdstDims,
                                            void *pdstPitches, void *pbatch,
                                            void *pbatchDims, void *pbatchPitches,
                                            unsigned int pbatchDimNum,
                                            unsigned int axis, float *scale,
                                            int32_t *offset, uint64_t flags);
void fwdLibSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize);
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize);
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize);
void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTy(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset);
void fwdLibIntLookupTableInstInt8QTy(void *dstT, void *dstDims,
                                     void *dstPitches, unsigned int dstDimNum,
                                     void *src1T, void *src1Dims,
                                     void *src1Pitches, void *src2T,
                                     void *src2Dims, void *src2Pitches);
void fwdLibFlushL3(uint32_t numShires);
/**********************
 * THREADED FUNCTIONS *
 **********************/
void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags);
void fwdLibIntLookupTableInstInt8QTyThreaded(
    void *dstT, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src1T, void *src1Dims, void *src1Pitches, void *src2T, void *src2Dims,
    void *src2Pitches, uint64_t flags);
void fwdLibBatchedAddInsti8i32Threaded(void *pdst, void *pdstDims, void *pdstPitches,
                               void *pbatch, void *pbatchDims,
                               void *pbatchPitches, unsigned int pbatchDimNum,
                               void *pslice, void *pslicePitches, float *scale,
                               int32_t *offset, uint64_t flags);
void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags);
/************************
 * VECTORIZED FUNCTIONS *
 ************************/
void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyVectorized(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags);
void fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyAligned32Bytes(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags);
template<typename DstType>
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdst2, void *pdst2Pitches, 
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, uint64_t flags,
    const uint32_t minionOffset = 0, const uint32_t numShires = 0);
template<typename DstType, bool Weighted = true>
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyOptimized(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, uint64_t flags,
    const uint32_t minionOffset = 0, const uint32_t numShires = 0);
template<typename DstType>
void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstFloatTyOptimized(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags,
    const uint32_t minionOffset = 0, const uint32_t numShires = 0);

} // namespace dnn_lib

#endif /* LIBNODES_H_ */
