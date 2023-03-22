/*-------------------------------------------------------------------------
 * Copyright (C) 2023, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include <array>
#include <stdio.h>

#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include <dnn_lib/Writer.h>
#include <dnn_lib/utils.h>
#include <dnn_lib/ETSOCGenericOpInst.h>
#include <dnn_lib/LibCommon.h>
#include <dnn_lib/LibTypes.h>
#include <dnn_lib/LibTensor.h>
#include <dnn_lib/Float16.h>


#include "entryPoint.h"
#include "kernel_arguments.h"

int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(1, entryPoint_0, nullptr);

using namespace dnn_lib;

int entryPoint_0(KernelArguments* kernelArgs) {

  et_assert(kernelArgs->nTensors == 2);

  auto inputDesc = &kernelArgs->tensors[0];
  auto outputDesc = &kernelArgs->tensors[1];

  // we need to hardcode dims for now to craete arrays for libtensor-dims.
  
  et_assert(inputDesc->nDims == 5);
  et_assert(outputDesc->nDims == 5);


  std::array<dim_t, 5> inputDims = {inputDesc->dims[0], inputDesc->dims[1], inputDesc->dims[2], inputDesc->dims[3], inputDesc->dims[4]};
  std::array<dim_t, 5> inputStrides =  {inputDesc->strides[0], inputDesc->strides[1], inputDesc->strides[2], inputDesc->strides[3], inputDesc->strides[4]};
  auto inputPtr = (void *) inputDesc->deviceAddress;
  constexpr auto inputElk = ElemKind::FloatTy;
  constexpr auto inputPaddingUntouchable  = false;
  LibTensor inputTensor(inputElk, inputPtr, inputDims, inputStrides, inputPaddingUntouchable);

  std::array<dim_t, 5> outputDims = {outputDesc->dims[0], outputDesc->dims[1], outputDesc->dims[2], outputDesc->dims[3], outputDesc->dims[4]};
  std::array<dim_t, 5> outputStrides = {outputDesc->strides[0], outputDesc->strides[1], outputDesc->strides[2], outputDesc->strides[3], outputDesc->strides[4]};
  auto outputPtr = (void *)  outputDesc->deviceAddress; 
  constexpr auto outputElk = ElemKind::FloatTy;
  constexpr auto outputPaddingUntouchable = false;
  LibTensor outputTensor(outputElk, outputPtr, outputDims, outputStrides, outputPaddingUntouchable);

  auto operation = (FFTOp) kernelArgs->operation;

  if (operation == FFTOp::SKIP) { // DEBUG op to skip the operation.
    return 0;
  }
  constexpr uint64_t flags = 31;
  constexpr uint32_t minionOffset = 0;
  constexpr uint32_t assignedMinions = 1024;

  
  uint32_t op = operation == FFTOp::FFT?  uint32_t(dnn_lib::inlining::Operation::FFT) : uint32_t(dnn_lib::inlining::Operation::IFFT);
  auto start = et_get_timestamp();
  
  dnn_lib::inlining::fwdLibETSOCGenericOpInst<inputElk>(&outputTensor, &inputTensor, op, flags, minionOffset, assignedMinions);

  auto elapsed = et_get_delta_timestamp(start);
 
  if(get_minion_id() == 0) {
    et_printf("%s %d fft took %lu Cycles\n", __func__,1,elapsed);
  }
 
  return 0;
}
