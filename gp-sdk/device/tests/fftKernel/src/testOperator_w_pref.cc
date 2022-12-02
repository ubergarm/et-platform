// clang-format off



#include <stdio.h>

#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include "testOperator_w_pref.h"
#include "kernel_arguments.h"
//#include "neuralizer_device_types.h"
#include "inst_pref_decls.h"


__attribute__((noinline))
_w_pref_inst_pref_sect_attr(testOperator)
void testOperator_w_pref(KernelArguments * layer_dyn_info) {

	// Inline the weight prefetch code of current node 
	testOperator_w_inst_pref_self(layer_dyn_info);


	// Inline the weight prefetch code of current node 
	testOperator_w_inst_pref(layer_dyn_info);

}
