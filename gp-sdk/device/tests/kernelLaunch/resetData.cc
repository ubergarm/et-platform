/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#include <array>
#include <stdio.h>
#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>


#include "entryPoint.h"

constexpr size_t size = 32ULL;
constexpr size_t smallSize = 4ULL;
class KernelArguments;
int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

static uint8_t smallData[smallSize] = {0,1,2,3}; // will be placed in .sdata section
static uint8_t initializedData[size] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}; // will be placed in .data section


/* This test checks if .data global region is properly reset
   on each kernel launch. 
   Current ETSOC-1 firmware copies the .data regions the first
   time the kernel (elf file) is loaded to the device, however,
   subsequent launches will not reset the data to the original values.
   
   In this test we modify .data and .sdata regions to check if the
   gp-sdk crt0 .data regions recovery system works properly */
int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  bool errorFound = 0;

  if (get_relative_thread_id() == 0) {

    // Check data is initialized correctly
    for(size_t i = 0; i < size; i++) {
      if (initializedData[i] != static_cast<uint8_t>(i)) {
        errorFound = 1;
        break;
      }
    }
    for(size_t i = 0; i < smallSize; i++) {
      if (smallData[i] != static_cast<uint8_t>(i)) {
        errorFound = 1;
        break;
      }
    }

    et_assert(!errorFound && "Error: .data section is not valid\n");

    // Add +1 to all .data region (initializedData and smallData)
    for(size_t i = 0; i < size; i++) {
      initializedData[i]++;
    }
    for(size_t i = 0; i < smallSize; i++) {
      smallData[i]++;
    }

    if (!errorFound) et_printf("Results are correct.");
  }
  
  return 0;
}
