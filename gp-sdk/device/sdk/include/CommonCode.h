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

/*! \file CommonCode.h
    \brief Efficient routines for common mem copy operations.
*/

// Global
#include <inttypes.h>
#include <etsoc/common/utils.h>

/**
 * Copies \p num_bytes bytes from the object pointed to by \p src to the object pointed to by \p dst. Both object
 * pointers must be 32-byte aligned. \brief Copies bytes from a memory position in the device to another. \param src
 * pointer to the memory location to copy from \param dst pointer to the memory location to copy to \param num_bytes
 * number of bytes to copy
 */
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

/*! \cond PRIVATE */

/**
 * Copies the static_cast<int>(\p value) repeatedly in 32-bit strides starting at \p ptr memory addres until \p
 * num_bytes are written. \brief Sets a region of memory to the same value
 *
 * \param ptr pointer to the memory location to copy to
 * \param value 32-bit value to write in memory
 * \param num_bytes number of bytes to write
 */
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
/*! \endcond */

#endif
