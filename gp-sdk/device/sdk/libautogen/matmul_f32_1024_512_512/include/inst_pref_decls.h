#ifndef INST_PREF_DECLS_H
#define INST_PREF_DECLS_H

#include "kernel_arguments.h"

inline uint64_t _min(uint64_t a, uint64_t b) { return (a > b)? a : b; }

#define _inst_pref_sect_attr(id)            __attribute__((section(#id "_inst_pref_sect")))
#define _inst_pref_inst_pref_sect_attr(id)  __attribute__((section(#id "_inst_pref_inst_pref_sect")))
#define _compute_inst_pref_sect_attr(id)    __attribute__((section(#id "_compute_inst_pref_sect")))
#define _act_pref_inst_pref_sect_attr(id)   __attribute__((section(#id "_act_pref_inst_pref_sect")))
#define _w_pref_inst_pref_sect_attr(id)     __attribute__((section(#id "_w_pref_inst_pref_sect")))

extern char _sync_compute_node_start;
extern char _sync_compute_node_end;
static const uint64_t log2_cache_line_bytes = 6;
static const uint64_t cache_line_bytes = 1 << log2_cache_line_bytes;
static const uint64_t cache_line_bytes_minus_one = cache_line_bytes - 1;
static const uint64_t function_prolog_bytes = 0;
static const uint64_t function_epilog_bytes = 0;
static const uint64_t max_prefetch_lines = 16;

extern "C" void MatmulCmd_inst_pref(kernelArguments * layer_dyn_info);
inline void MatmulCmd_w_inst_pref_self(kernelArguments * layer_dyn_info) {

	uint64_t worker_id;
	uint64_t self_size;
	uint64_t self_ptr;
	uint64_t self_ptr_end;
	uint64_t ret_ptr;
	uint64_t hart_id = get_hart_id();
	uint64_t minion_id = (hart_id >> 1) & 0x1F;
	// Map the minion id from the 16..19 and 24..27 ranges to 0..7
	if (minion_id >= 24) {
	  worker_id = minion_id - 20;
	} else {
	  worker_id = minion_id - 16;
	}

  //defining clobber  x1 to reg x1;
  //defining clobber  x3 to reg x31;
// Declaring registers: a total of 4
	__asm__ __volatile__ (
	 //Skip for minion other than zero
	"bne  %[worker_id], zero, 1f\n"
	 //Stride is one cacheline
	"addi  x31, zero, 0x40 /*64*/ \n"
	"la  %[self_ptr], __start_MatmulCmd_w_pref_inst_pref_sect\n"
	 //Intending to read, prefetch destination being L2, size 4 lines
	"li  x1, 0x400000000000003 /*288230376151711747*/ \n"
	 //Self prefetch ASAP
	"or  x1, x1, %[self_ptr]\n"
	 //Issue prefetch
	"csrw  prefetch_va /* 0x81f */, x1\n"
	"1:\n"
      : 
        [self_ptr] "=r" (self_ptr) 
      : 
        [worker_id] "r" (worker_id) 
      : 
       /*x1*/"x1",
       /*x3*/"x31" 
	);

  //defining clobber  x1 to reg x1;
  //defining clobber  x3 to reg x31;
// Declaring registers: a total of 5
	__asm__ __volatile__ (
	"la  %[self_ptr], __start_MatmulCmd_w_pref_inst_pref_sect\n"
	"la  %[self_ptr_end], __stop_MatmulCmd_w_pref_inst_pref_sect\n"
      : 
        [self_ptr] "=r" (self_ptr),
        [self_ptr_end] "=r" (self_ptr_end) 
      : 
        [worker_id] "r" (worker_id) 
      : 
       /*x1*/"x1",
       /*x3*/"x31" 
	);

	uint64_t startCL = ((self_ptr+ cache_line_bytes_minus_one) >> log2_cache_line_bytes) +4;
	uint64_t endCL = (self_ptr_end+ cache_line_bytes_minus_one + function_epilog_bytes) >> log2_cache_line_bytes;
	if(endCL>=startCL) {
	   self_size = endCL - startCL + 1 ;
	   self_ptr += (4 << log2_cache_line_bytes) ;
	// Load balance the prefetch work accross the 8 minions
	{
	  uint64_t extra_work = self_size & 7;
	  self_size = self_size >> 3;
	  uint64_t minion_offset_lines = worker_id * self_size;
	  if (worker_id < extra_work) {
	    minion_offset_lines += worker_id;
	    ++self_size;
	  }
	  else {
	    minion_offset_lines += extra_work;
	  }
	  self_ptr += minion_offset_lines * cache_line_bytes;
	}

	  int pending_lines = self_size;
	  for( ;  pending_lines > 0 ; pending_lines -=16){
	    self_size = (pending_lines > 16 ? 16:pending_lines)-1;
  //defining clobber  x1 to reg x1;
  //defining clobber  x3 to reg x3;
  //defining clobber  x31 to reg x31;
// Declaring registers: a total of 7
	__asm__ __volatile__ (
	 //Intending to read, prefetch destination being L2
	"li  x1, 0x400000000000000 /*288230376151711744*/ \n"
	 // Stride is one cacheline
	"addi  x31, zero, 0x40 /*64*/ \n"
	 //Weight prefetcher: pointer and size
	"or  x3, x1, %[self_ptr]\n"
	"or  x3, x3, %[self_size]\n"
	 //Issue a prefetch
	"csrw  prefetch_va /* 0x81f */, x3\n"
      : 
        [self_ptr_end] "=r" (self_ptr_end) 
      : 
        [worker_id] "r" (worker_id),
        [self_ptr] "r" (self_ptr),
        [self_size] "r" (self_size) 
      : 
       /*x1*/"x1",
       /*x3*/"x3",
       /*x31*/"x31" 
	);

	    //advance 16 CL
	    self_ptr+=(16*64);
	  } //end of for
	}//end of if(endCL>startCL)
  //defining clobber  x1 to reg x1;
  //defining clobber  x3 to reg x3;
  //defining clobber  x31 to reg x31;
// Declaring registers: a total of 8

	__asm__ __volatile__ (
	 //Skip for minion other than zero
	"bne  %[worker_id], zero, 1f\n"
	 //Intending to read, prefetch destination being L2, size 3 lines
	"li  x1, 0x400000000000002 /*288230376151711746*/ \n"
	 //Stride is one cacheline
	"addi  x31, zero, 0x40 /*64*/ \n"
	 //Prefetch the return code
	"la  %[ret_ptr], MatmulCmd_w_pref_return_point\n"
	"andi  %[ret_ptr], %[ret_ptr], 0xffffffffffffffc0 /*-64*/ \n"
	"or  x3, x1, %[ret_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
	"1:\n"
      : 
        [self_ptr_end] "=r" (self_ptr_end),
        [ret_ptr] "=r" (ret_ptr) 
      : 
        [self_ptr] "r" (self_ptr),
        [self_size] "r" (self_size),
        [worker_id] "r" (worker_id) 
      : 
       /*x1*/"x1",
       /*x3*/"x3",
       /*x31*/"x31" 
	);

}
inline void MatmulCmd_w_inst_pref(kernelArguments * layer_dyn_info) {

	uint64_t worker_id;
	uint64_t ret_ptr;
  //defining clobber  x1 to reg x1;
  //defining clobber  x3 to reg x3;
  //defining clobber  x31 to reg x31;
// Declaring registers: a total of 5

	__asm__ __volatile__ (
	 //Intending to read, prefetch destination being L2, size 3 lines
	"li  x1, 0x400000000000002 /*288230376151711746*/ \n"
	 //Stride is one cacheline
	"addi  x31, zero, 0x40 /*64*/ \n"
	 //Prefetch the return code
	"la  %[ret_ptr], MatmulCmd_w_pref_return_point\n"
	"andi  %[ret_ptr], %[ret_ptr], 0xffffffffffffffc0 /*-64*/ \n"
	"or  x3, x1, %[ret_ptr]\n"
	"csrw  prefetch_va /* 0x81f */, x3\n"
      : 
        [ret_ptr] "=r" (ret_ptr) 
      : 
        [worker_id] "r" (worker_id) 
      : 
       /*x1*/"x1",
       /*x3*/"x3",
       /*x31*/"x31" 
	);

}
#endif // INST_PREF_DECLS_H
