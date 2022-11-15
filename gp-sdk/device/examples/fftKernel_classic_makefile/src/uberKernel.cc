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

// clang-format off



#include <stdio.h>

#include <etsoc/isa/hart.h>
#include <etsoc/isa/barriers.h>
#include <etsoc/isa/cacheops-umode.h>
#include <etsoc/common/utils.h>
#include "kernel_arguments.h"
#include "testOperator_compute.h"

extern "C" void startUberKernelComputes(uint64_t shireId, uint32_t minionId, kernelArguments * layer_dyn_info);


extern "C" int uberKernel_RAWKERNEL_entry_point(kernelArguments * layer_dyn_info) {
  uint32_t hart = get_hart_id();
  uint32_t threadId = hart & 1;
  uint32_t minionId = hart >> 1;
  uint32_t shireId = minionId >> 5;
  minionId = minionId & 0x1F;
  if ((shireId < 32) and (threadId == 0)) {
    startUberKernelComputes(shireId,minionId,layer_dyn_info);
  } 
  return 0;
} 

#define _UNIQUE_CALL(function, ...)\
 do { \
  function( __VA_ARGS__ ); \
  __asm__ __volatile__ (\
    ".global " #function "_return_point\n"\
    #function "_return_point:" : : : );\
} while(0);


extern "C" __attribute((noclone , noinline)) void startUberKernelComputes(uint64_t shireId, uint32_t minionId, kernelArguments * layer_dyn_info){

  _UNIQUE_CALL(testOperator_compute, layer_dyn_info)
}
