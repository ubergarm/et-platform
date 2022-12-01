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
  if (get_minion_id()==0) {
    et_printf("%s,%d HELLO WORLD!!!!\n",__func__,__LINE__);
  }

  return 0;
}
