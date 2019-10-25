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

#ifndef __LIBCOMMON_H
#define __LIBCOMMON_H

#include "cacheops.h"

//-------------------------------------------------------------------------------------------------
//
// FUNCTION: evict_va_multi
//
//   This function is a wrapper of evict_va for any number for cache lines. It calls evict_va as
//   many times as needed to evict all lines
//
inline void __attribute__((always_inline)) evict_va_multi(uint64_t dst, uintptr_t addr, uint64_t num_lines) {
  while (num_lines > 16) {
    evict_va(0, dst, addr, 15, 64);
    addr += 64;
    num_lines -= 16;
  }
  if (num_lines > 0)
    evict_va(0, dst, addr, num_lines-1, 64);
}

#endif // ! __LIBCOMMON_H
