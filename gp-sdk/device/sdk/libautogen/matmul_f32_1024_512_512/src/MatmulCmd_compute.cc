
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

#define ACTIVE_T0_MASK2	0xffffffffULL //Mask of minions that have thread 0 active
#define ACTIVE_T0_M12	0x1f //Minions that have thread 0 active minus 1
#define ACTIVE_SH_MASK	0xffffffffULL //Mask of active shires
#define ACTIVE_T0_MASK	0xffffffffULL //Mask of minions that have thread 0 active
#define ACTIVE_T0_M1	0x1f //Minions that have thread 0 active minus 1
#include <stdio.h>

#include <etsoc/isa/hart.h>


#include <etsoc/isa/esr_defines.h>
#include <dnn_lib/utils.h>

#include "kernel_arguments.h"

#include "inst_pref_decls.h"
#include "MatmulCmd_compute.h"
#include "MatmulCmd_act_pref.h"
#include "MatmulCmd_w_pref.h"


_compute_inst_pref_sect_attr(MatmulCmd)
__attribute__((noinline))
void MatmulCmd_compute(kernelArguments * layer_dyn_info) {

  //Start of the function 
  __asm__ __volatile__ (
	"mov.m.x  m0, zero, 0xff /*255*/ \n"
  :
  :
  :
  );
  // Gets the minion id
  uint64_t hart_id = get_hart_id();
  uint64_t minion_id_orig = (hart_id >> 1) & 0x1F;
  uint64_t minion_id = minion_id_orig ;
  uint64_t shire_id_orig = hart_id >> 6;
  uint64_t shire_id = shire_id_orig ;
  // Wait the credit from the empty act_pref
  fcc(FCC_0); 
  //Define The Pitches of each Tensor 
  uint64_t tensor_act0_pitch_dim0 = 2048;
  uint64_t tensor_act0_pitch_dim1 = 4;
  uint64_t tensor_out0_pitch_dim0 = 2048;
  uint64_t tensor_out0_pitch_dim1 = 4;
  uint64_t tensor_weight0_pitch_dim0 = 2112;
  uint64_t tensor_weight0_pitch_dim1 = 4;
  // Gets the IPI address to thread0
  uint64_t barrier_o_t = (31 << 5) + 0;
  { //start of NON turn-off minion code 
  //defining clobber  temp_reg0 to reg x31;
  //defining clobber  temp_reg1 to reg x30;
  //defining clobber  temp_reg2 to reg x29;
uint64_t minion_id_dim_0 = (( minion_id >> 0 )& 0x7) ;
uint64_t minion_id_dim_1 = (( minion_id >> 3 )& 0x3) ;
uint64_t minion_id_dim_2 = (( minion_id >> 3 )& 0x0) ;
uint64_t shire_id_dim_0 = (( shire_id  >> 0 )& 0x3);
uint64_t shire_id_dim_1 = (( shire_id  >> 2 )& 0x7);
uint64_t shire_id_dim_2 = (( shire_id  >> 2 )& 0x0);
  //defining clobber  loopIDim2 to reg x3;
  //defining clobber  loopIDim1 to reg x4;
uint64_t coopMaskA = ((/*CoopID*/  + 0) | (((/*coopMaskMinion*/  (( (0xff << ( minion_id & 0x18)) >> ( minion_id & 0x18 )) & 0xFF)   )   )<< 8 ) | (((/*coopMaskNeigh*/ (0x1<<(minion_id >> 3)) & 0xff) ) << 16 ));
uint64_t coopMaskB = ((/*CoopID*/ (minion_id & 0x7) + 1) | (((/*coopMaskMinion*/  (( (0x1010101 << ( minion_id & 0x7)) >> ( minion_id & 0x18 )) & 0xFF)   )   )<< 8 ) | (((/*coopMaskNeigh*/ (0xF) & 0xff) ) << 16 ));
uint64_t aRowInit_id0Minion = ((minion_id_dim_1 * 32)) /*ARow*/;
uint64_t aRowInit_id0 = ((shire_id_dim_1 * 128)) + aRowInit_id0Minion /*ARow*/;
uint64_t aColInit_id0Minion = (0) /*ACol*/;
uint64_t aColInit_id0 = (0) + aColInit_id0Minion /*ACol*/;
uint64_t tensor_act0_init_id0 = (uint64_t) layer_dyn_info->placeholder_0 + 0x0 + (aRowInit_id0 * tensor_act0_pitch_dim0) + (aColInit_id0 * tensor_act0_pitch_dim1);
  //defining clobber  tensor_act0_dyn_id0 to reg x7;
uint64_t bRowInit_id0Minion = ( (0)>>0) /*BRow*/;
uint64_t bRowInit_id0 = ( (0)>>0) + bRowInit_id0Minion /*BRow*/;
uint64_t bColInit_id0Minion = ((minion_id_dim_0 * 16)) /*BCol*/;
uint64_t bColInit_id0 = ((shire_id_dim_0 * 128)) + bColInit_id0Minion /*BCol*/;
uint64_t tensor_weight0_init_id0 = (uint64_t) layer_dyn_info->constant_0 + 0x0 + (bColInit_id0 * tensor_weight0_pitch_dim1) + (bRowInit_id0 * tensor_weight0_pitch_dim0);
  //defining clobber  tensor_weight0_dyn_id0 to reg x5;
uint64_t tensor_out0_dyn_id0 = (uint64_t) layer_dyn_info->placeholder_0 + 0x202000 + aRowInit_id0 * tensor_out0_pitch_dim0 + bColInit_id0 * tensor_out0_pitch_dim1;
  //defining clobber  tensor_fma to reg x28;
// Declaring registers: a total of 13
     __asm__ __volatile__ (

	"li  /*tensor_act0_dyn_id0*/ x7, 0x400000000000000f /*4611686018427387919*/ \n"
	"li  /*tensor_weight0_dyn_id0*/ x5, 0x421000000000000f /*4760304806130614287*/ \n"
	"add  /*tensor_weight0_dyn_id0*/ x5, /*tensor_weight0_dyn_id0*/ x5, %[tensor_weight0_init_id0]\n"
	"add  /*tensor_act0_dyn_id0*/ x7, /*tensor_act0_dyn_id0*/ x7, %[tensor_act0_init_id0]\n"
	 //scaleCodeSCPL1 0  0
	"li  /*tensor_fma*/ x28, 0x1ff800000100001 /*143974450588549121*/ \n"
	 //adjustTileConfigInit1
	 //adjustTileConfigInit2
	 //adjustTileConfigInit0
	 //Open loop  for dimension 1 ARow nIters=2
	 //Reset Loop counter of Dim:  1
	"li  /*loopIDim1*/ x4, 0x2 /*2*/ \n"
	"loop_dim_1:\n"
	 //Open loop  for dimension 2 ACol nIters=32
	 //Reset Loop counter of Dim:  2
	"li  /*loopIDim2*/ x3, 0x20 /*32*/ \n"
	".p2alignw 2,0x0001\n"
	"loop_dim_2:\n"
	 //        Set A TensorLoad Pitch

	"li  x31, 0x800 /*2048*/ \n"
	 //        Execute A TensorLoad

	"csrw  tensor_coop /* 0x804 */, %[coopMaskA]\n"
	"csrw  tensor_load /* 0x83f */, /*tensor_act0_dyn_id0*/ x7\n"
	 //        Set B TensorLoad Pitch

	"li  x31, 0x841 /*2113*/ \n"
	 //        Execute B TensorLoad

	"csrw  tensor_coop /* 0x804 */, %[coopMaskB]\n"
	"csrw  tensor_load /* 0x83f */, /*tensor_weight0_dyn_id0*/ x5\n"
	"compute_tile_result_loop:\n"
	 //Wait for Tload A to be done 
	"csrw  tensor_wait /* 0x830 */, 0x0 /*0*/ \n"
	 //        Start TensorFMA

	"csrw  tensor_fma /* 0x801 */, /*tensor_fma*/ x28\n"
	 //        Clear the Start flag 

	"andi  /*tensor_fma*/ x28, /*tensor_fma*/ x28, 0xfffffffffffffffe /*-2*/ \n"
	 //Switch registers for tensorload A
	"li  /*temp_reg0*/ x31, 0x200000000000000 /*144115188075855872*/ \n"
	"xor  /*tensor_act0_dyn_id0*/ x7, /*tensor_act0_dyn_id0*/ x7, /*temp_reg0*/ x31\n"
	 //Switch registers for tensorFMA
	"xori  /*tensor_fma*/ x28, /*tensor_fma*/ x28, 0x100 /*256*/ \n"
	 //---  End of loop for dimension 2 ACol----
	 //-----------------------------------------------------
	"addi  /*loopIDim2*/ x3, /*loopIDim2*/ x3, 0xffffffffffffffff /*-1*/ \n"
	"beq  /*loopIDim2*/ x3, zero, exit_loop_dim_2\n"
	 //Advance In
	"addi  /*tensor_act0_dyn_id0*/ x7, /*tensor_act0_dyn_id0*/ x7, 0x40 /*64*/ \n"
	 //Advance weights 33792
	"li  /*temp_reg0*/ x31, 0x8400 /*33792*/ \n"
	"add  /*tensor_weight0_dyn_id0*/ x5, /*tensor_weight0_dyn_id0*/ x5, /*temp_reg0*/ x31\n"
	"j  loop_dim_2\n"
	"exit_loop_dim_2:\n"
	 //--- Finish End of loop for dimension 2 ACol----
	 //-----------------------------------------------------
	 //Check if we have to write results
	"addi  /*loopIDim1*/ x4, /*loopIDim1*/ x4, 0xffffffffffffffff /*-1*/ \n"
	"beq  /*loopIDim1*/ x4, zero, write_results\n"
	 //Update data for next iteration of the loop
	 //Advance a tensor a full row, less the previously advanced
	"li  /*temp_reg0*/ x31, 0x7840 /*30784*/ \n"
	"add  /*tensor_act0_dyn_id0*/ x7, /*tensor_act0_dyn_id0*/ x7, /*temp_reg0*/ x31\n"
	 //Reset the position of the weights
	"li  /*temp_reg0*/ x31, 0xfffffffffff00400 /*-1047552*/ \n"
	"add  /*tensor_weight0_dyn_id0*/ x5, /*tensor_weight0_dyn_id0*/ x5, /*temp_reg0*/ x31\n"
	 //        Set A TensorLoad Pitch

	"li  x31, 0x800 /*2048*/ \n"
	 //        Execute A TensorLoad

	"csrw  tensor_coop /* 0x804 */, %[coopMaskA]\n"
	"csrw  tensor_load /* 0x83f */, /*tensor_act0_dyn_id0*/ x7\n"
	 //        Set B TensorLoad Pitch

	"li  x31, 0x841 /*2113*/ \n"
	 //        Execute B TensorLoad

	"csrw  tensor_coop /* 0x804 */, %[coopMaskB]\n"
	"csrw  tensor_load /* 0x83f */, /*tensor_weight0_dyn_id0*/ x5\n"
	 //Mark the next tensorFMA as 1st one, so it must clear the registers
	"ori  /*tensor_fma*/ x28, /*tensor_fma*/ x28, 0x1 /*1*/ \n"
	"write_results:\n"
	 //storeChunkSize:  64 nMaxStores:  1 n16BxRow before: 4 resultsDependent: 0 nStoresToPerform: 1 nStoresSkip: 0
	 //tensorStore cfg: incShift 0 offsetReg 0 n16BxRow 4 getTensorStoreCoopMode() 0 getTensorStoreDelay() 0 nRows 16 PSDD.getBiggerPackage() 16 jitterData_.getStoreChunkSize() 64 bColSizeDependOnIter 0 bColSizeDependOnMinion 0 PSDD.getDiffMinionPackage() 0
	"li  /*temp_reg1*/ x30, 0x1f8000000000000 /*141863388262170624*/ \n"
	"add  /*temp_reg1*/ x30, /*temp_reg1*/ x30, %[tensor_out0_dyn_id0]\n"
	 // stride for store is 2048
	"li  x31, 0x800 /*2048*/ \n"
	"csrw  tensor_store /* 0x87f */, /*temp_reg1*/ x30\n"
	"skip_write_results:\n"
	 //If last result, we are done
	"beq  /*loopIDim1*/ x4, zero, compute_done_a_rows\n"
	 //Reset Loop counter of Dim:  2
	"li  /*loopIDim2*/ x3, 0x20 /*32*/ \n"
	 //afterWrite: Advance Out Tensor 
	"li  /*temp_reg0*/ x31, 0x8000 /*32768*/ \n"
	"add  %[tensor_out0_dyn_id0], %[tensor_out0_dyn_id0], /*temp_reg0*/ x31\n"
	 //jump in the middle of the first loop, because some tensor load has already been done
	"j  compute_tile_result_loop\n"
	"compute_done_a_rows:\n"
      : 
        [tensor_out0_dyn_id0] "+&r" (tensor_out0_dyn_id0) 
      : 
        [coopMaskA] "r" (coopMaskA),
        [coopMaskB] "r" (coopMaskB),
        [tensor_act0_init_id0] "r" (tensor_act0_init_id0),
        [tensor_weight0_init_id0] "r" (tensor_weight0_init_id0) 
      : 
       /*temp_reg0*/"x31",
       /*temp_reg1*/"x30",
       /*temp_reg2*/"x29",
       /*loopIDim2*/"x3",
       /*loopIDim1*/"x4",
       /*tensor_act0_dyn_id0*/"x7",
       /*tensor_weight0_dyn_id0*/"x5",
       /*tensor_fma*/"x28" 
    
);

  //End of the function 
} //end of NON turn-off minion code 
    }
