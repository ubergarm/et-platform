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

#include <array>
#include <stdio.h>

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>

#include "entryPoint.h"
class KernelArguments;
int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

static void __attribute__((noinline)) forcebt4() {
    for(;;) {}
}

static void __attribute__((noinline)) forcebt3() {
    forcebt4();
}
static void __attribute__((noinline)) forcebt2() {
    forcebt3();
}
static void __attribute__((noinline)) forcebt1() {
    forcebt2();
}

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  
  forcebt1();
  
  return 0;
}
