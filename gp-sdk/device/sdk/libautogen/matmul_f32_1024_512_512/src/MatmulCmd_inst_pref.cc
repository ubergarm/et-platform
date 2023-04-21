#include <stdio.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/esr_defines.h>
#include "kernel_arguments.h"

#include "inst_pref_decls.h"

__attribute((noinline))
_inst_pref_sect_attr(MatmulCmd)
void MatmulCmd_inst_pref(kernelArguments * layer_dyn_info) {

uint64_t self_ptr;
uint64_t act_pref_ptr;
uint64_t act_pref_size;
uint64_t act_pref_ret_ptr;
uint64_t compute_ptr;
uint64_t compute_size;
uint64_t compute_ret_ptr;
uint64_t sync_ptr;
uint64_t sync_size;
uint64_t max_prefetch_lines;
uint64_t ret_ptr;
  //defining clobber  temp_reg0 to reg x1;
  //defining clobber  temp_reg1 to reg x4;
  //defining clobber  temp_reg2 to reg x3;
  //defining clobber  temp_reg3 to reg x31;
// Declaring registers: a total of 15

	__asm__ __volatile__ (
	 //Stride is one cacheline
	"addi  x31, zero, 0x40 /*64*/ \n"
	 //Intending to read, prefetch destination being L2, pre-substract one
	"li  x1, 0x3ffffffffffffff /*288230376151711743*/ \n"
	 //Self prefetch itself, but skip the first line
	"la  %[self_ptr], __start_MatmulCmd_inst_pref_sect+64\n"
	"addi  x3, x1, 0x4 /*4*/ \n"
	"or  x3, x3, %[self_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
	 //Activation prefetch prefetcher (a few lines to prefetch ASAP)
	"addi  x4, x1, 0x4 /*4*/ \n"
	"la  %[act_pref_ptr], __start_MatmulCmd_act_pref_inst_pref_sect\n"
	"or  x3, x4, %[act_pref_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
	 //Compute prefetcher (a few lines to prefetch ASAP)
	"la  %[compute_ptr], __start_MatmulCmd_compute_inst_pref_sect\n"
	"or  x3, x4, %[compute_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
	 //Setup max_prefetch_lines
	"li  %[max_prefetch_lines], 0x10 /*16*/ \n"
	 //Prefetching from act_pref_ptr
	"addi  %[act_pref_ptr], %[act_pref_ptr], 0x100 /*256*/ \n"
	"1:\n"
	"la  %[act_pref_size], __stop_MatmulCmd_act_pref_inst_pref_sect\n"
	"addi  %[act_pref_size], %[act_pref_size], 0x3f /*63*/ \n"
	"srli  %[act_pref_size], %[act_pref_size], 0x6 /*6*/ \n"
	"slli  %[act_pref_size], %[act_pref_size], 0x6 /*6*/ \n"
	"bleu  %[act_pref_size], %[act_pref_ptr], 3f\n"
	"sub  %[act_pref_size], %[act_pref_size], %[act_pref_ptr]\n"
	"srli  %[act_pref_size], %[act_pref_size], 0x6 /*6*/ \n"
	"bleu  %[act_pref_size], %[max_prefetch_lines], 2f\n"
	"li  %[act_pref_size], 0x10 /*16*/ \n"
	"2:\n"
	"add  x3, x1, %[act_pref_size]\n"
	"or  x3, x3, %[act_pref_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
	"slli  %[act_pref_size], %[act_pref_size], 0x6 /*6*/ \n"
	"add  %[act_pref_ptr], %[act_pref_ptr], %[act_pref_size]\n"
	"j  1b\n"
	"3:\n"
	 //Prefetching from compute_ptr
	"addi  %[compute_ptr], %[compute_ptr], 0x100 /*256*/ \n"
	"1:\n"
	"la  %[compute_size], __stop_MatmulCmd_compute_inst_pref_sect\n"
	"addi  %[compute_size], %[compute_size], 0x3f /*63*/ \n"
	"srli  %[compute_size], %[compute_size], 0x6 /*6*/ \n"
	"slli  %[compute_size], %[compute_size], 0x6 /*6*/ \n"
	"bleu  %[compute_size], %[compute_ptr], 3f\n"
	"sub  %[compute_size], %[compute_size], %[compute_ptr]\n"
	"srli  %[compute_size], %[compute_size], 0x6 /*6*/ \n"
	"bleu  %[compute_size], %[max_prefetch_lines], 2f\n"
	"li  %[compute_size], 0x10 /*16*/ \n"
	"2:\n"
	"add  x3, x1, %[compute_size]\n"
	"or  x3, x3, %[compute_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
	"slli  %[compute_size], %[compute_size], 0x6 /*6*/ \n"
	"add  %[compute_ptr], %[compute_ptr], %[compute_size]\n"
	"j  1b\n"
	"3:\n"
	 //Prefetch sync_compute_node
	// "la  %[sync_ptr], _sync_compute_node_start\n"
	"la  %[sync_ptr], __start_sync_compute_node\n"
	"addi  x3, x1, 0x5 /*5*/ \n"
	"or  x3, x3, %[sync_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
	 //Set prefetch size to 3 lines
	"addi  x1, x1, 0x3 /*3*/ \n"
	 //Prefetch return code
	"la  %[ret_ptr], MatmulCmd_inst_pref_return_point\n"
	"andi  %[ret_ptr], %[ret_ptr], 0xffffffffffffffc0 /*-64*/ \n"
	"or  x3, x1, %[ret_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
	 //Prefetch act_pref return code
	"la  %[act_pref_ret_ptr], MatmulCmd_act_pref_return_point\n"
	"andi  %[act_pref_ret_ptr], %[act_pref_ret_ptr], 0xffffffffffffffc0 /*-64*/ \n"
	"or  x3, x1, %[act_pref_ret_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
	 //Prefetch compute return code
	"la  %[compute_ret_ptr], MatmulCmd_compute_return_point\n"
	"andi  %[compute_ret_ptr], %[compute_ret_ptr], 0xffffffffffffffc0 /*-64*/ \n"
	"or  x3, x1, %[compute_ret_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
      : 
        [self_ptr] "=r" (self_ptr),
        [act_pref_ptr] "=r" (act_pref_ptr),
        [act_pref_size] "=r" (act_pref_size),
        [act_pref_ret_ptr] "=r" (act_pref_ret_ptr),
        [compute_ptr] "=r" (compute_ptr),
        [compute_size] "=r" (compute_size),
        [compute_ret_ptr] "=r" (compute_ret_ptr),
        [sync_ptr] "=r" (sync_ptr),
        [sync_size] "=r" (sync_size),
        [max_prefetch_lines] "=r" (max_prefetch_lines),
        [ret_ptr] "=r" (ret_ptr) 
      : 
      : 
       /*temp_reg0*/"x1",
       /*temp_reg1*/"x4",
       /*temp_reg2*/"x3",
       /*temp_reg3*/"x31" 
);
}
