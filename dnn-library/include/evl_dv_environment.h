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

#ifndef EVL_DV_ENVIRONMENT_H
#define EVL_DV_ENVIRONMENT_H

#include <inttypes.h>

#define NUM_MINIONS_PER_NEIGH (8)
#define NUM_NEIGH_PER_SHIRE   (4)
#define NUM_HARTS_PER_MINION  (2)

#define L1_WAYS     4
#define L1_SETS     16
#define L1_CL_SIZE  64

#define ET_DIAG_NOP             (0x0)
#define ET_DIAG_PUTCHAR         (0x1)
#define ET_DIAG_RAND            (0x2)
#define ET_DIAG_RAND_MEM_UPPER  (0x3)
#define ET_DIAG_RAND_MEM_LOWER  (0x4)
#define ET_IRQ_INJ              (0x5)
#define ET_DIAG_ECC_INJ         (0x6)
#define ET_DIAG_CYCLE           (0x7)

//
// Utility functions
//
inline void __attribute__((always_inline)) et_diag_command(uint8_t command, uint64_t payload)
{
   uint64_t csr_enc = ((uint64_t) command << 56) |
                      (payload & 0x00FFFFFFFFFFFFFFULL);
   __asm__ __volatile__ (
         "csrw validation1, %[csr_enc]\n"
         :
         : [csr_enc] "r" (csr_enc)
         :
   );
}

inline void __attribute__((always_inline)) et_write_val2(uint64_t val)
{
   __asm__ __volatile__ (
         "csrw validation2, %[csr_enc]\n"
         :
         : [csr_enc] "r" (val)
         :
   );
}

//
// Print to STDOUT
//
inline void __attribute__((always_inline)) et_putchar(char c)
{
   et_diag_command(ET_DIAG_PUTCHAR, (uint64_t) c);
}
inline void __attribute__((always_inline)) et_printf(const char *str)
{
   while (*str) {
      et_putchar(*str);
      str++;
   }
}

//
// Random number generation
//
inline uint32_t __attribute__((always_inline)) et_get_rand_word()
{
   uint64_t reg;
   et_diag_command(ET_DIAG_RAND, 0);
   __asm__ __volatile__ ( "csrr %[n], validation1\n\t" :[n] "=r" (reg) : :);
   return (uint32_t) reg;
}

// Return a random value >= than 'lower' and <= than 'upper'
inline uint32_t __attribute__((always_inline)) et_get_rand_word(uint32_t lower, uint32_t upper)
{
   uint64_t reg;
   et_write_val2(((uint64_t) upper << 32) | ((uint64_t) lower));
   et_diag_command(ET_DIAG_RAND, 1);
   __asm__ __volatile__ ( "csrr %[n], validation1\n\t" :[n] "=r" (reg) : :);
   return (uint32_t) reg;
}

inline uint64_t __attribute__((always_inline)) et_get_rand_dword()
{
   uint64_t reg;
   et_diag_command(ET_DIAG_RAND, 2);
   __asm__ __volatile__ ( "csrr %[n], validation1\n\t" :[n] "=r" (reg) : :);
   return reg;
}

//
// Get cycle count
//
inline uint64_t __attribute__((always_inline)) et_get_cycle_count()
{
   uint64_t reg;
   et_diag_command(ET_DIAG_CYCLE, 0);
   __asm__ __volatile__ ( "csrr %[n], validation1\n\t" :[n] "=r" (reg) : :);
   return reg;
}

//
// Memory randomization
//
inline void __attribute__((always_inline)) et_val_rand_mem(uint64_t addr, uint32_t sub_cmd)
{
   et_diag_command(ET_DIAG_RAND_MEM_LOWER, ((uint64_t) (addr & 0x00000000FFFFFFFF)) | ((uint64_t) sub_cmd << 32));
   et_diag_command(ET_DIAG_RAND_MEM_UPPER, ((uint64_t) (addr & 0xFFFFFFFF00000000)) | ((uint64_t) sub_cmd << 32));
}

#endif // ! EVL_DV_ENVIRONMENT_H
