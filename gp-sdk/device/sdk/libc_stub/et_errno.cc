/*-------------------------------------------------------------------------
 * Copyright (C) 2023, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
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
