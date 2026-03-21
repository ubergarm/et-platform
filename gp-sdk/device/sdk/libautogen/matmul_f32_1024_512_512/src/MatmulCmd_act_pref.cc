
/*-------------------------------------------------------------------------
 *
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 * 
 ************************************************************
 * --------------------------------------------------------- 
 * This code is Auto-Generated. Please DON'T MODIFY IT. 
 * --------------------------------------------------------- 
 ************************************************************
 *------------------------------------------------------------------------- 
 */ 

#include <stdio.h>
#include <etsoc/isa/hart.h>


#include <etsoc/isa/esr_defines.h>
#include <dnn_lib/utils.h>
#include "kernel_arguments.h"
#include "neuralizer_device_types.h"
#include "MatmulCmd_act_pref.h"
#include "inst_pref_decls.h"




__attribute((noinline))
_act_pref_inst_pref_sect_attr(MatmulCmd)
void MatmulCmd_act_pref(kernelArguments * layer_dyn_info) {

  //Start of the function 
  // Gets the minion id
  uint64_t hart_id = get_hart_id();
  uint64_t shire_id_orig = hart_id >> 6;
  uint64_t shire_id = shire_id_orig ;
  uint64_t minion_id_orig = (hart_id >> 1) & 0x1F;
  if(minion_id_orig == 0){
    fcc_send(shire_id, THREAD_0, FCC_0, 0xffffffff); 
  }

  //End of the function 
}
