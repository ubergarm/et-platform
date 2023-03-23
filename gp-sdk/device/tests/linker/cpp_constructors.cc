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
#include "etsoc/isa/atomic.h"

#include "entryPoint.h"

class KernelArguments;
int entryPoint(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint, entryPoint);

uint32_t global_value1_with_constructor = 1;
uint32_t global_value2_with_constructor = 2;

// Classes to test global object contructors.
struct TestClass1 {
  __attribute__((noinline)) TestClass1() {
    atomic_or_global_32(&global_value1_with_constructor, 0x100000);
    atomic_or_global_32(&global_value2_with_constructor, 0x100000);
  }
  ~TestClass1() {
  }
};

struct TestClass2 {
  TestClass2() {
    atomic_or_global_32(&global_value1_with_constructor, 0x200);
    atomic_or_global_32(&global_value2_with_constructor, 0x200);
  }
  ~TestClass2() {
  }
};

static TestClass1 globalObj1;
static TestClass2 globalObj2;

int entryPoint([[maybe_unused]] KernelArguments* args) {
  et_assert(global_value1_with_constructor == (1 | 0x200 | 0x100000));
  et_assert(global_value2_with_constructor == (2 | 0x200 | 0x100000));
  return 0;
}
