/*-------------------------------------------------------------------------
 * Copyright (C) 2023, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _COMMON_CODE_H_
#define _COMMON_CODE_H_

// Global
#include <inttypes.h>
#include <etsoc/common/utils.h>



static inline int global_memcpy(void * dst, const void * src, size_t num_bytes) {
  constexpr size_t stride = 32; // vector width is 32 bytes
  /* cast to 1-byte ptr type, needed for pointer arithmetic */
  uint8_t *d = static_cast<uint8_t *>(dst);
  const uint8_t *s = static_cast<const uint8_t *>(src);

  /* Check 32 byte alignment */
  bool aligned = (reinterpret_cast<uintptr_t>(s) % 32UL == 0) &&
                 (reinterpret_cast<uintptr_t>(d) % 32UL == 0);

  et_assert(aligned  && "src/dst pointers are not 32-byte aligned");

  float tmp;
  for (size_t i = 0; i < num_bytes; i += stride) {
    __asm__ __volatile__("flwg.ps %[tmp], (%[src])\n"
                         "fswg.ps %[tmp], (%[dst])\n"
                         : [ tmp ] "=&f" (tmp)
                         : [ src ] "r" (s + i),
                           [ dst ] "r"(d + i)
                         :);
  }
  return 0;
}

static inline int global_memset(void * ptr, const int value, size_t num_bytes) {
  /* vector width is 32 bytes (256-bit) */
  constexpr size_t stride = 32; 
  /* cast to 1 byte type, enables pointer arithmetic */
  uint8_t *p = static_cast<uint8_t *>(ptr);

  /* Check 32 byte alignment */
  bool aligned = (reinterpret_cast<uintptr_t>(p) % 32UL == 0);
  // et_assert(aligned && "pointer is not 32-byte aligned");

  if (aligned) {
    float valueVector;
    __asm__ __volatile__("fbcx.ps %[valueVector], %[value]\n"
                        : [ valueVector ] "=&f" (valueVector)
                        : [ value ] "r" (value)
                        :);

    for (size_t i = 0; i < num_bytes; i += stride) {
      __asm__ __volatile__( "fswg.ps %[valueVector], (%[ptr])\n"
                            :  
                            : [ ptr ] "r"(p + i), [ valueVector ] "f" (valueVector)
                            :);
    }
  }
  return 0;
}

#endif
