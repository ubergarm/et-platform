/*-------------------------------------------------------------------------
 * Copyright (C) 2020, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _DNNLIB_INLINING_ALL_H_
#define _DNNLIB_INLINING_ALL_H_

#include "AdaptiveAvgPoolInst.h" // From include/inlining path
#include "ArgMaxInst.h" // From include/inlining path
#include "AvgPoolInst.h" // From include/inlining path
#include "BatchedAddInst.h" // From include/inlining path
#include "BatchedReduceAddInst.h" // From include/inlining path
#include "BatchOneHotInst.h" // From include/inlining path
#include "Checksum.h" // From include/inlining path
#include "ConvertToInst.h" // From include/inlining path
#include "Convolution3DInst.h" // From include/inlining path
#include "ConvolutionInst.h" // From include/inlining path
#include "CopyInst.h" // From include/inlining path
#include "CopyInstTensorized.h" // From include/inlining path
#include "CrossEntropyLossInst.h" // From include/inlining path
#include "DequantizeInst.h" // From include/inlining path
#include "ElementBoolInst.h" // From include/inlining path
#include "ElementExpInst.h" // From include/inlining path
#include "ElementImmLogic.h" // From include/inlining path
#include "ElementInst.h" // From include/inlining path
#include "ElementIsNaNInst.h" // From include/inlining path
#include "ElementSelectInst.h" // From include/inlining path
#include "ElementSingleInst.h" // From include/inlining path
#include "EmbeddingBagInst.h" // From include/inlining path
#include "ExtractTensorInst.h" // From include/inlining path
#include "FlipInst.h" // From include/inlining path
#include "FullyConnectedInst.h" // From include/inlining path
#include "FusedRowwiseQuantizedSparseLengthsSum.h" // From include/inlining path error:template overload fix when finish
#include "FusedRowwiseQuantizedSparseLengthsWeightedSum.h" // From include/inlining path  error:template overload fix when finish
#include "GatherInst.h" // From include/inlining path
#include "GatherRangesInst.h" // From include/inlining path
#include "InsertTensorInst.h" // From include/inlining path
#include "IntLookupTableInst.h" // From include/inlining path
#include "LengthsSumInst.h" // From include/inlining path
#include "LengthsToRangesInst.h" // From include/inlining path
#include "LocalResponseNormalizationInst.h" // From include/inlining path
#include "MatMulInst.h" // From include/inlining path
#include "MaxPoolInst.h" // From include/inlining path
#include "MaxSplatInst.h" // From include/inlining path
#include "ModuloInst.h" // From include/inlining path
#include "QuantizeInst.h" // From include/inlining path
#include "RescaleQuantizedInst.h" // From include/inlining path
#include "RowwiseQuantizedFullyConnected.h" // From include/inlining path
#include "RowwiseQuantizedSparseLengthsWeightedSum.h" // From include/inlining path
#include "ScatterDataInst.h" // From include/inlining path
#include "SigmoidInst.h" // From include/inlining path
#include "SoftMaxInst.h" // From include/inlining path
#include "SparseLengthsWeightedSumInst.h" // From include/inlining path
#include "SparseToDenseInst.h" // From include/inlining path
#include "SparseToDenseMaskInst.h" // From include/inlining path
#include "SplatInst.h" // From include/inlining path
#include "SyncopyInstTensorized.h" // From include/inlining path
#include "TanhInst.h" // From include/inlining path
#include "TensorViewInst.h" // From include/inlining path
#include "TopKInst.h" // From include/inlining path
#include "TransposeInst.h" // From include/inlining path

#define dispatchLibImplEltWiseSingle(functionName, pm1, op, ...) \
  dnn_lib::inlining::functionName<pm1, op>(__VA_ARGS__)

#define dispatchLibImplEltWise(functionName, pm1, pm2, op, ...) \
  dnn_lib::inlining::functionName<pm1, pm2, op>(__VA_ARGS__)

#define dispatchLibImplEltWiseParal(functionName, pm1, pm2, pm3, op, ...) \
  dnn_lib::inlining::functionName<pm1, pm2, pm3, op>(__VA_ARGS__)

#define dispatchLibImpl2Types(functionName, pm1, pm2, ...) \
  dnn_lib::inlining::functionName<pm1, pm2>(__VA_ARGS__)

#define dispatchLibImpl3Types(functionName, pm1, pm2, pm3, ...) \
  dnn_lib::inlining::functionName<pm1, pm2, pm3>(__VA_ARGS__)

#define dispatchLibImpl(functionName, pm, ...) \
  dnn_lib::inlining::functionName<pm>(__VA_ARGS__)

#define dispatchLibNonInliningImpl(functionName, pm, ...) \
  dnn_lib::functionName<pm>(__VA_ARGS__)

#define dispatchLibQuantizedTyImpl(functionName, pm, ...) \
  dnn_lib::inlining::functionName<pm>(__VA_ARGS__)

#define dispatchLibWithIndexImpl(functionName, pm1, pm2, ...) \
  dnn_lib::inlining::functionName<pm1, pm2>(__VA_ARGS__)

#define dispatchLibConvertImpl(functionName, pm1, pm2, ...) \
  dnn_lib::inlining::functionName<pm1, pm2>(__VA_ARGS__)

#define dispatchLibIntTyImpl(functionName, pm, ...) \
  dnn_lib::inlining::functionName<pm>(__VA_ARGS__)

#endif
