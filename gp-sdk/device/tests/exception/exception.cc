// clang-format off

#include <array>
#include <stdio.h>

#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include "entryPoint.h"

__attribute__((noinline))
int entryPoint([[maybe_unused]] KernelArguments * args) {
  
  //Generate code exception
  *(volatile uint64_t *)0 = 0xDEADBEEF; 

  return 0;
}
