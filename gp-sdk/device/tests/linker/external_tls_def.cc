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
