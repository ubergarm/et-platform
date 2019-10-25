/*-------------------------------------------------------------------------
* Copyright (C) 2018, Esperanto Technologies Inc.
* The copyright to the computer program(s) herein is the
* property of Esperanto Technologies, Inc. All Rights Reserved.
* The program(s) may be used and/or copied only with
* the written permission of Esperanto Technologies and
* in accordance with the terms and conditions stipulated in the
* agreement/contract under which the program(s) have been supplied.
*-------------------------------------------------------------------------
*/

#ifndef __MACROS
#define __MACROS

#ifdef __ASSEMBLER__ // when included in an assembly file

#define ASM_TEST_START \
   fence	   ;\
   lui  a7, 0xDEAD0; \
   csrw validation0, a7;

#define ASM_TEST_PASS \
   fence	   ;\
   lui a7, 0x1FEED; \
   csrw validation0, a7; \
   1:wfi; \
   j 1b; \

#define ASM_TEST_PASS_CSTALL \
   fence	   ;\
   lui a7, 0x1FEED; \
   csrw validation0, a7; \
   csrw stall, zero; \

#define ASM_TEST_FAIL \
   fence	   ;\
   lui a7, 0x50BAD; \
   csrw validation0, a7; \
   wfi; \

#define ASM_TEST_FAIL_CSTALL \
   fence	   ;\
   lui a7, 0x50BAD; \
   csrw validation0, a7; \
   csrw stall, zero; \

#define ASM_TEST_PASS_MAX \
   fence	   ;\
   li a0, 2;\
   li a1, 0;\
   jal tb_write;\
   wfi; \

#define FENCE fence;

#define WAIT_TENSOR_LOAD_0     csrwi 0x830, 0;
#define WAIT_TENSOR_LOAD_1     csrwi 0x830, 1;
#define WAIT_TENSOR_LOAD_L2_0  csrwi 0x830, 2;
#define WAIT_TENSOR_LOAD_L2_1  csrwi 0x830, 3;
#define WAIT_PREFETCH_0        csrwi 0x830, 4;
#define WAIT_PREFETCH_1        csrwi 0x830, 5;
#define WAIT_CACHEOPS          csrwi 0x830, 6;
#define WAIT_TENSOR_FMA        csrwi 0x830, 7;
#define WAIT_TENSOR_STORE      csrwi 0x830, 8;
#define WAIT_TENSOR_REDUCE     csrwi 0x830, 9;
#define WAIT_TENSOR_QUANT      csrwi 0x830, 10;

#else // __ASSEMBLER__ --> when included in a C/C++ file

#define C_TEST_START \
   __asm__ __volatile__ ( \
         "fence\n"\
         "lui  a7, 0xDEAD0\n" \
         "csrw validation0, a7\n" \
         : : : "a7");

#define C_TEST_PASS \
   __asm__ __volatile__ ( \
         "fence\n" \
         "lui a7, 0x1FEED\n" \
         "csrw validation0, a7\n" \
         "1:wfi\n" \
         "j 1b\n" \
         : : : "a7");

#define C_TEST_FAIL \
   __asm__ __volatile__ ( \
         "fence\n"\
         "lui a7, 0x50BAD\n" \
         "csrw validation0, a7\n" \
         "wfi\n" \
         : : : "a7");

#define NOP  __asm__ __volatile__ ("nop\n");
#define FENCE __asm__ __volatile__ ("fence\n");
#define WFI __asm__ __volatile__ ("wfi\n");
#define WAIT_TENSOR_LOAD_0     __asm__ __volatile__ ( "csrwi 0x830, 0\n" : : );
#define WAIT_TENSOR_LOAD_1     __asm__ __volatile__ ( "csrwi 0x830, 1\n" : : );
#define WAIT_TENSOR_LOAD_L2_0  __asm__ __volatile__ ( "csrwi 0x830, 2\n" : : );
#define WAIT_TENSOR_LOAD_L2_1  __asm__ __volatile__ ( "csrwi 0x830, 3\n" : : );
#define WAIT_PREFETCH_0        __asm__ __volatile__ ( "csrwi 0x830, 4\n" : : );
#define WAIT_PREFETCH_1        __asm__ __volatile__ ( "csrwi 0x830, 5\n" : : );
#define WAIT_CACHEOPS          __asm__ __volatile__ ( "csrwi 0x830, 6\n" : : );
#define WAIT_TENSOR_FMA        __asm__ __volatile__ ( "csrwi 0x830, 7\n" : : );
#define WAIT_TENSOR_STORE      __asm__ __volatile__ ( "csrwi 0x830, 8\n" : : );
#define WAIT_TENSOR_REDUCE     __asm__ __volatile__ ( "csrwi 0x830, 9\n" : : );
#define WAIT_TENSOR_QUANT      __asm__ __volatile__ ( "csrwi 0x830, 10\n" : : );
#define STALL                  __asm__ __volatile__ ( "csrw stall, x0\n" : : );
#define CLEAR_TENSOR_ERROR     __asm__ __volatile__ ( "csrwi 0x808, 0" : : );

static inline __attribute__((always_inline)) uintptr_t ecall(uintptr_t arg0, uintptr_t arg1, uintptr_t arg2, uintptr_t arg3)
{
    register uintptr_t a0 asm ("a0") = arg0;
    register uintptr_t a1 asm ("a1") = arg1;
    register uintptr_t a2 asm ("a2") = arg2;
    register uintptr_t a3 asm ("a3") = arg3;
    __asm__ __volatile__ (
        "ecall"
        : "+r" (a0)
        : "r" (a1), "r" (a2), "r" (a3)
        : "memory");
    return a0;
}

#endif // __ASSEMBLER__

#endif // ! __MACROS
