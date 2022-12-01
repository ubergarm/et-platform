// clang-format off


#include <array>
#include <stdio.h>
#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include "entryPoint.h"

constexpr size_t size = 256ULL;

static uint8_t uninitializedData[size];
static uint8_t initToZeroData[size] = {0};

__attribute__((noinline))
int entryPoint([[maybe_unused]] KernelArguments * args) {
  bool errorFound = 0;
  if (get_minion_id()==0) {
    for(size_t i = 0; i < size; i++) {
      if (uninitializedData[i] != 0 || initToZeroData[i] != 0) {
        errorFound = 1;
        break;
      }
    }

    et_assert(!errorFound && "Error: .bss section is not set to 0\n");

    if (!errorFound) et_printf("Results are correct.");
  }
  
  return 0;
}
