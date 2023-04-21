
/*-------------------------------------------------------------------------
 * Copyright (C) 2018, Esperanto Technologies Inc. 
 * 
 ************************************************************
 * --------------------------------------------------------- 
 * This code is Auto-Generated. Please DON'T MODIFY IT. 
 * --------------------------------------------------------- 
 ************************************************************
 * 
 * The copyright to the computer program(s) herein is the 
 * property of Esperanto Technologies, Inc. All Rights Reserved. All Rights Reserved. 
 * The program(s) may be used and/or copied only with  
 * the written permission of Esperanto Technologies and 
 * in accordance with the terms and conditions stipulated in the 
 * agreement/contract under which the program(s) have been supplied. 
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
