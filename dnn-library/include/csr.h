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

 

#ifndef __CSR_H
#define __CSR_H

inline void __attribute__((always_inline)) csr_write(uint16_t addr, uint64_t val)
{
    __asm__ __volatile__(
        "csrw %[addr],%[val]\n"
        :
        : [addr] "i" (addr), [val] "r" (val)
        :
    );
}

inline uint64_t __attribute__((always_inline)) csr_read(uint16_t addr)
{
  uint64_t ret;
   __asm__ __volatile__ ("csrr %[ret], %[addr]\n"
                         : [ret] "=r" (ret)
                         : [addr] "i" (addr)
                         :
  );
  return ret;
}






#endif // ! __CSR_H
