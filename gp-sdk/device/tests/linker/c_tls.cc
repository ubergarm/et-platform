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

__thread uint32_t testTls1;

static void __attribute__((noinline)) testTlsVarSimple() {
  // Dump of assembler code for function testTlsVars():
  //   0x0000000000001f00 <+0>:	mv	a5,tp
  //   0x0000000000001f04 <+4>:	lw	a4,0(a5)
  //   0x0000000000001f08 <+8>:	addiw	a4,a4,1
  //   0x0000000000001f0c <+12>:	sw	a4,0(a5)
  //   0x0000000000001f10 <+16>:	ret
  testTls1++;
}

__thread uint32_t testTls2 = 1;

static void __attribute__((noinline)) testTlsVarInitialized() {
  testTls2++;
  auto hartId = get_hart_id();
  testTls2 += hartId;

  et_assert(testTls2 == 1 + 1 + hartId);
}

__thread uint32_t testTls3 = 0;

static void __attribute__((noinline)) testTlsVarZeroed() {
  testTls3++;
  auto hartId = get_hart_id();
  testTls3 += hartId;

  et_assert(testTls3 == 1 + hartId);
}

static void __attribute__((noinline)) testTlsVarStatic() {
  static __thread uint32_t testTls4;
  auto hartId = get_hart_id();
  testTls4++;
  testTls4 += hartId;
  et_assert(testTls4 == 1 + hartId);
}

static void __attribute__((noinline)) testTlsVarStatic_1() {
  static thread_local uint32_t testTls5;
  auto hartId = get_hart_id();
  testTls5++;
  testTls5 += hartId;
  et_assert(testTls5 == 1 + hartId);
}
int entryPoint([[maybe_unused]] KernelArguments* args) {
  testTlsVarSimple();
  testTlsVarInitialized();
  testTlsVarZeroed();
  testTlsVarStatic();
  testTlsVarStatic_1();
  return 0;
}
