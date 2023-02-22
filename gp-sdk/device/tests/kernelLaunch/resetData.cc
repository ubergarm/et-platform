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


#include "entryPoint.h"

constexpr size_t size = 32ULL;
constexpr size_t smallSize = 4ULL;

int entryPoint_0(KernelArguments* args);
DeviceConfig config {1, entryPoint_0, nullptr};
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

  if (get_minion_id() == 0) {

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
