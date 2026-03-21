/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 * -------------------------------------------------------------------------
 */ 

// ET-SoC-1 errno stub. 
// Note, this implementation relies on TLS enabled on linker script & crt0 
#include <errno.h>

extern "C" {
extern int *__errno (void) {
  static __thread int thread_local_errno;
  return &thread_local_errno;
}
}
