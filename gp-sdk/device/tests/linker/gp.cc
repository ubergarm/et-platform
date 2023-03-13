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

static volatile int testBigData[1024 * 1024] = {1};
static volatile int testMedData[96] = {1};
static volatile int testSmallData[4] = {1, 2, 3, 4};
static volatile int testVerySmallData[1] = {1};
static volatile char testVeryVerySmallData[1] = {'a'};
void testBig();
void testMed();
void testSmall();
void testVerySmall();
void testVeryVerySmall();

// There is no self-test for this at the moment.
// Expected result is whatever fits in .sdata will use linker-relaxation and gp-based access.
// (as per default small definition, this is testVerySmallData and testVeryVerySmall data).
// see https://www.sifive.com/blog/all-aboard-part-3-linker-relaxation-in-riscv-toolchain
//
// (gdb) disassemble  testVeryVerySmall
// Dump of assembler code for function _Z17testVeryVerySmallv:
//   0x0000000000002100 <+0>:	addi	sp,sp,-16
//   0x0000000000002104 <+4>:	auipc	a5,0x3
//   0x0000000000002108 <+8>:	lbu	a5,-700(a5) # 0x4e48 <_ZL21testVeryVerySmallData>
//   0x000000000000210c <+12>:	sb	a5,15(sp)
//   0x0000000000002110 <+16>:	lbu	a5,15(sp)
//   0x0000000000002114 <+20>:	addi	sp,sp,16
//   0x0000000000002118 <+24>:  ret

int entryPoint(KernelArguments* args);
extern constexpr __DeviceConfig config{2, entryPoint, entryPoint};

void __attribute__((noinline)) testBig() {
  volatile auto dummy = testBigData[0];
  (void)dummy;
}

void __attribute__((noinline)) testMed() {
  volatile auto dummy = testMedData[0];
  (void)dummy;
}

void __attribute__((noinline)) testSmall() {
  volatile auto dummy = testSmallData[0];
  (void)dummy;
}

void __attribute__((noinline)) testVerySmall() {
  volatile auto dummy = testVerySmallData[0];
  (void)dummy;
}

void __attribute__((noinline)) testVeryVerySmall() {
  volatile auto dummy = testVeryVerySmallData[0];
  (void)dummy;
}

int entryPoint([[maybe_unused]] KernelArguments* args) {
  if (get_hart_id() != 0) {
    return 0;
  }

  testBig();
  testMed();
  testSmall();
  testVerySmall();
  testVeryVerySmall();

  return 0;
}
