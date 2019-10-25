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

#ifndef __MINION_H
#define __MINION_H

#ifdef __cplusplus
extern "C"
{
  #include <cstdint>
#endif

#ifdef __cplusplus
}
#endif


typedef volatile __attribute__((aligned(64))) union {
   uint8_t  b[64];
   uint16_t h[32];
   uint32_t w[16];
   uint64_t d[8];
} minion_cache_line_t;


typedef volatile __attribute__((aligned(4096))) struct {
   minion_cache_line_t cl[64];
} minion_dcache_t;


inline unsigned int __attribute__((always_inline)) get_hart_id()
{
  unsigned int ret;
  __asm__ __volatile__ (
      "csrr %[ret], hartid\n"
    : [ret] "=r" (ret)
  );
  return ret;
}

inline unsigned int __attribute__((always_inline)) get_shire_id()
{
   return get_hart_id() >> 6;
}

inline unsigned int __attribute__((always_inline)) get_neigh_id()
{
  return (get_hart_id() >> 4) & 3;
}

inline unsigned int __attribute__((always_inline)) get_minion_id()
{
   return get_hart_id() >> 1;
}

inline unsigned int __attribute__((always_inline)) get_thread_id()
{
   return get_hart_id() & 1;
}

inline uint64_t __attribute__((always_inline)) wait_for_credits(uint64_t credit_counter) {
  uint64_t result;
  __asm__ __volatile__ (
      "csrrw  %[result], 0x821 ,%[credit_counter]\n"
    : [result] "=r" (result)
    : [credit_counter] "r" (credit_counter)
  );
  return result;
}

#endif // ! __MINION_H
