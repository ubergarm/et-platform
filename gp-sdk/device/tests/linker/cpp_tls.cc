/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>

#include "entryPoint.h"

class KernelArguments;
int entryPoint(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint, entryPoint);

struct CppTls {
  static inline thread_local uint32_t value = 1;
  static inline thread_local uint32_t otherValue = 0;
  CppTls() {
    value++;
    value += get_hart_id();
    otherValue++;
  }
  void foo() {
    static thread_local uint32_t var;
    var++;
    var += get_hart_id();
    // assuming only called once.
    et_assert(var == 1 + get_hart_id());
  }
};

int entryPoint([[maybe_unused]] KernelArguments* args) {
  CppTls testCppTls;
  et_assert(testCppTls.value == 1 + 1 + get_hart_id());
  et_assert(testCppTls.otherValue == 1);
  testCppTls.foo();
  return 0;
}
