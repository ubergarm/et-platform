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

#include "AdaptiveAvgPoolInst.h"
#include "ArgMaxInst.h"
#include "AvgPoolInst.h"
#include "BatchOneHotInst.h"
#include "BatchedAddInst.h"
#include "BatchedReduceAddInst.h"
#include "BatchedReduceMinInst.h"
#include "ChannelWiseQuantizedConvolutionInst.h"
#include "ConvTransposeInst.h"
#include "ConvertToInst.h"
#include "Convolution3DInst.h"
#include "ConvolutionInst.h"
#include "CopyInst.h"
#include "CrossEntropyLossInst.h"
#include "CumSumInst.h"
#include "Dequantize4BitsColumnBlocksInst.h"
#include "Dequantize8BitsColumnBlocksInst.h"
#include "DequantizeInst.h"
#include "ElementBinaryInst.h"
#include "ElementBoolInst.h"
#include "ElementImmLogic.h"
#include "ElementSelectInst.h"
#include "ElementSingleInst.h"
#include "EmbeddingBagInst.h"
#include "ExtractTensorInst.h"
#include "FlipInst.h"
#include "FullyConnectedInst.h"
#include "FusedRowwiseQuantizedSparseLengthsSumInst.h"         // error:template overload fix when finish
#include "FusedRowwiseQuantizedSparseLengthsWeightedSumInst.h" // error:template overload fix when finish
#include "GatherInst.h"
#include "GatherRangesInst.h"
#include "InsertTensorInst.h"
#include "IntLookupTableInst.h"
#include "LengthsRangeFillInst.h"
#include "LengthsSumInst.h"
#include "LengthsToRangesInst.h"
#include "LocalResponseNormalizationInst.h"
#include "MatMulInst.h"
#include "MaxPoolInst.h"
#include "MaxSplatInst.h"
#include "ModuloInst.h"
#include "NonMaxSuppressionInst.h"
#include "OneHotInst.h"
#include "ProfileInst.h"
#include "QuantizeInst.h"
#include "RescaleQuantizedInst.h"
#include "ResizeBilinearInst.h"
#include "ResizeNearestInst.h"
#include "RowwiseQuantizedFullyConnectedInst.h"
#include "RowwiseQuantizedSparseLengthsWeightedSumInst.h"
#include "ScatterDataInst.h"
#include "SoftMaxInst.h"
#include "SpaceToDepthInst.h"
#include "SparseLengthsSumInst.h"
#include "SparseLengthsWeightedSumInst.h"
#include "SparseToDenseInst.h"
#include "SparseToDenseMaskInst.h"
#include "SplatInst.h"
#include "SyncopyInst.h"
#include "TensorViewInst.h"
#include "TopKInst.h"
#include "TransposeInst.h"
#include "TriluInst.h"

#define dispatchLibImplEltWiseSingle(functionName, pm1, op, ...) dnn_lib::inlining::functionName<pm1, op>(__VA_ARGS__)

#define dispatchLibImplEltWise(functionName, pm1, pm2, op, ...)                                                        \
  dnn_lib::inlining::functionName<pm1, pm2, op>(__VA_ARGS__)

#define dispatchLibImplEltWiseParal(functionName, pm1, pm2, pm3, op, ...)                                              \
  dnn_lib::inlining::functionName<pm1, pm2, pm3, op>(__VA_ARGS__)

#define dispatchLibImpl2Types(functionName, pm1, pm2, ...) dnn_lib::inlining::functionName<pm1, pm2>(__VA_ARGS__)

#define dispatchLibImpl3Types(functionName, pm1, pm2, pm3, ...)                                                        \
  dnn_lib::inlining::functionName<pm1, pm2, pm3>(__VA_ARGS__)

#define dispatchLibImpl(functionName, pm, ...) dnn_lib::inlining::functionName<pm>(__VA_ARGS__)

#define dispatchLibNonInliningImpl(functionName, pm, ...) dnn_lib::functionName<pm>(__VA_ARGS__)

#define dispatchLibQuantizedTyImpl(functionName, pm, ...) dnn_lib::inlining::functionName<pm>(__VA_ARGS__)

#define dispatchLibWithIndexImpl(functionName, pm1, pm2, ...) dnn_lib::inlining::functionName<pm1, pm2>(__VA_ARGS__)

#define dispatchLibConvertImpl(functionName, pm1, pm2, ...) dnn_lib::inlining::functionName<pm1, pm2>(__VA_ARGS__)

#define dispatchLibIntTyImpl(functionName, pm, ...) dnn_lib::inlining::functionName<pm>(__VA_ARGS__)

#endif
