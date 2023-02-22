// clang-format off

#include <array>
#include <stdio.h>

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>

#include "entryPoint.h"

int entryPoint_0(KernelArguments* args);
DeviceConfig config {1, entryPoint_0, nullptr};

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  
  //Generate code exception
  *(volatile uint64_t *)0 = 0xDEADBEEF; 

  return 0;
}
