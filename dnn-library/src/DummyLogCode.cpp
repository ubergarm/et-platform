/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
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
