
/*-------------------------------------------------------------------------
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
#include "MatmulCmd_w_pref.h"
#include "inst_pref_decls.h"




__attribute((noinline))
_w_pref_inst_pref_sect_attr(MatmulCmd)
void MatmulCmd_w_pref(kernelArguments * layer_dyn_info) {

  //Start of the function 

  // Inline the weight prefetch code of the own code 

  // (only first node will have a non empty) 
  MatmulCmd_w_inst_pref_self(layer_dyn_info);

  // Gets the minion id
  uint64_t hart_id = get_hart_id();
  uint64_t shire_id_orig = hart_id >> 6;
  uint64_t shire_id = shire_id_orig ;
  uint64_t minion_id_orig = (hart_id >> 1) & 0x1F;

  //End of the function 

  // Inline the weight prefetch code of the next node 
  MatmulCmd_w_inst_pref(layer_dyn_info);

}
