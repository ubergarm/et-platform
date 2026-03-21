/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 * -------------------------------------------------------------------------
 */

#ifdef __clang__
extern "C" {
int atexit(void (*function)(void)) {
  (void) function;
  return 0;
}
}
#endif
