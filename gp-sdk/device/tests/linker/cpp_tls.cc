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

int entryPoint(KernelArguments* args);
extern const DeviceConfig config{2, entryPoint, entryPoint};

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
