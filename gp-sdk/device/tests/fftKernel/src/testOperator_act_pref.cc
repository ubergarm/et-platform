// clang-format off


#include <stdio.h>

#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include "testOperator_act_pref.h"
#include "kernel_arguments.h"
//#include "neuralizer_device_types.h"
#include "inst_pref_decls.h"


__attribute__((noinline))
_act_pref_inst_pref_sect_attr(testOperator)
void testOperator_act_pref(KernelArguments * layer_dyn_info) {
	 uint64_t hart_id = get_hart_id();
	 uint64_t shire_id = hart_id >> 6;
	 uint64_t minion_id = (hart_id >> 1) & 0x1F;
	 if(minion_id == 0){
	  fcc_send(shire_id, THREAD_0, FCC_0, 0xffffffff); 
	 }
}
