/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

// placeholder code to allow dnnLib standalone build.

#include <cstdint>

void log_enter_user_region_wrapper(uint64_t ptr, uint16_t regionId);
void log_exit_user_region_wrapper(uint64_t ptr, uint16_t regionId);
void log_enter_user_region_wrapper(uint64_t ptr, uint16_t regionId) {
  (void)ptr;
  (void)regionId;
  /* void */
}
void log_exit_user_region_wrapper(uint64_t ptr, uint16_t regionId) {
  (void)ptr;
  (void)regionId;
  /* void */
}
