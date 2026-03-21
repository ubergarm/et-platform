/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>

#include "entryPoint.h"

int entryPoint(void * args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint, entryPoint);

const thread_local uint32_t  * testTlsPtr;
const uint32_t glblRoData[10]={0};

int entryPoint([[maybe_unused]] void * args) {
  testTlsPtr = &glblRoData[0];
  extern void testExternalTlsAccess();
  testExternalTlsAccess();
  return 0;
}
