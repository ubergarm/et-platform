// clang-format off


#include <array>
#include <stdio.h>
//#include <device_common.h>
#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include "SyncComputeNode.h"
#include "inst_pref_decls.h"

#include "entryPoint.h"

#include "kernel_arguments.h"


typedef int8_t i8;

constexpr size_t size = 256ULL;

static uint8_t uninitializedData[size];
static uint8_t initToZeroData[size] = {0};

__attribute__((noinline))
int entryPoint(kernelArguments * layer_dyn_info) {
  bool errorFound = 0;
  if (get_minion_id()==0) {
    for(size_t i = 0; i < size; i++) {
      if (uninitializedData[i] != 0 || initToZeroData[i] != 0) {
        errorFound = 1;
        break;
      }
    }

    if (errorFound) {
      et_printf("Error: .bss is different than zero\n");
    }
    else {
      et_printf(".bss initialized correctly\n");
    }
  }
  
  return 0;
}
