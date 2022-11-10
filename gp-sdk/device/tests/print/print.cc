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


__attribute__((noinline))
int entryPoint(kernelArguments * layer_dyn_info) {

  //fcc(FCC_0);

  if (get_minion_id()==0) {
    et_printf("%s,%d HELLO WORLD!!!!\n",__func__,__LINE__);
    et_assert(false);
  }

  return 0;
}
