
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

#ifndef _MATMULCMD_COMPUTE_H
#define _MATMULCMD_COMPUTE_H

//Includes
#include "neuralizer_device_types.h"
#include "kernel_arguments.h"

//Defines
//Functions
extern "C" void MatmulCmd_compute(kernelArguments * layer_dyn_info);
extern "C" void MatmulCmd_act_pref(kernelArguments * layer_dyn_info);
#endif
