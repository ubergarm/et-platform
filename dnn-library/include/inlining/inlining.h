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

//TODO: RESTORE #include "AdaptiveAvgPoolInst.h" // From include/inlining path
//TODO: RESTORE #include "ArgMaxInst.h" // From include/inlining path
//TODO: RESTORE #include "AvgPoolInst.h" // From include/inlining path
//TODO: RESTORE #include "BatchOneHotInst.h" // From include/inlining path
//TODO: RESTORE #include "BatchedAddInst.h" // From include/inlining path
//TODO: RESTORE #include "BatchedReduceAddInst.h" // From include/inlining path
//TODO: RESTORE #include "Checksum.h" // From include/inlining path
#include "ConvertToInst.h" // From include/inlining path
//TODO: RESTORE #include "Convolution3DInst.h" // From include/inlining path
//TODO: RESTORE #include "ConvolutionInst.h" // From include/inlining path
//TODO: RESTORE #include "CopyInst.h" // From include/inlining path
//TODO: RESTORE #include "CopyInstTensorized.h" // From include/inlining path
//TODO: RESTORE #include "CrossEntropyLossInst.h" // From include/inlining path
//TODO: RESTORE #include "DequantizeInst.h" // From include/inlining path
//TODO: RESTORE #include "ETSOCMaxSplatInst.h" // From include/inlining path
//TODO: RESTORE #include "ElementBoolInst.h" // From include/inlining path
//TODO: RESTORE #include "ElementExpInst.h" // From include/inlining path
//TODO: RESTORE #include "ElementImmLogic.h" // From include/inlining path
//TODO: RESTORE #include "ElementInst.h" // From include/inlining path
//TODO: RESTORE #include "ElementIsNaNInst.h" // From include/inlining path
//TODO: RESTORE #include "ElementSelectInst.h" // From include/inlining path
//TODO: RESTORE #include "ElementSingleInst.h" // From include/inlining path
//TODO: RESTORE #include "EmbeddingBagInst.h" // From include/inlining path
//TODO: RESTORE #include "ExtractTensorInst.h" // From include/inlining path
//TODO: RESTORE #include "FlipInst.h" // From include/inlining path
//TODO: RESTORE #include "FullyConnectedInst.h" // From include/inlining path
//TODO: RESTORE #include "FusedRowwiseQuantizedSparseLengthsWeightedSum.h" // From include/inlining path
//TODO: RESTORE #include "FusedRowwiseQuantizedSparseLengthsSum.h" // From include/inlining path
//TODO: RESTORE #include "GatherInst.h" // From include/inlining path
//TODO: RESTORE #include "GatherRangesInst.h" // From include/inlining path
//TODO: RESTORE #include "InsertTensorInst.h" // From include/inlining path
//TODO: RESTORE #include "IntLookupTableInst.h" // From include/inlining path
//TODO: RESTORE #include "LengthsSumInst.h" // From include/inlining path
//TODO: RESTORE #include "LengthsToRangesInst.h" // From include/inlining path
//TODO: RESTORE #include "LocalResponseNormalizationInst.h" // From include/inlining path
//TODO: RESTORE #include "MatMulInst.h" // From include/inlining path
//TODO: RESTORE #include "MatMulInstTransposed.h" // From include/inlining path
//TODO: RESTORE #include "MaxPoolInst.h" // From include/inlining path
//TODO: RESTORE #include "ModuloInst.h" // From include/inlining path
//TODO: RESTORE #include "QuantizeInst.h" // From include/inlining path
//TODO: RESTORE #include "RescaleQuantizedInst.h" // From include/inlining path
//TODO: RESTORE #include "RowwiseQuantizedFullyConnected.h" // From include/inlining path
//TODO: RESTORE #include "RowwiseQuantizedSparseLengthsWeightedSum.h" // From include/inlining path
//TODO: RESTORE #include "ScatterDataInst.h" // From include/inlining path
//TODO: RESTORE #include "SigmoidInst.h" // From include/inlining path
//TODO: RESTORE #include "SoftMaxInst.h" // From include/inlining path
//TODO: RESTORE #include "SparseLengthsWeightedSumInst.h" // From include/inlining path
//TODO: RESTORE #include "SparseToDenseInst.h" // From include/inlining path
//TODO: RESTORE #include "SparseToDenseMaskInst.h" // From include/inlining path
//TODO: RESTORE #include "SplatInst.h" // From include/inlining path
//TODO: RESTORE #include "SyncopyInstTensorized.h" // From include/inlining path
//TODO: RESTORE #include "TanhInst.h" // From include/inlining path
//TODO: RESTORE #include "TensorViewInst.h" // From include/inlining path
//TODO: RESTORE #include "TopKInst.h" // From include/inlining path
//TODO: RESTORE #include "TransposeInst.h" // From include/inlining path

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
