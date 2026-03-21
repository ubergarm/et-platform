/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#include <array>
#include <stdio.h>

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>

#include "entryPoint.h"

#pragma GCC push_options
#pragma GCC optimize ("-O0")

class KernelArguments;
int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

template <class Tp> inline __attribute__((always_inline)) void DoNotOptimize(Tp& value) {
#if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

static int __attribute__((noinline,optnone)) forcebt4(int level) {
  volatile int level_4 = level * 2;
  DoNotOptimize(level_4);
  for (;;) {
  }
  return level_4;
}

static int __attribute__((noinline,optnone)) forcebt3(int level) {
  volatile int level_3 = level * 2;
  DoNotOptimize(level_3);
  level_3 = forcebt4(level_3);
  return level_3;
}

static int __attribute__((noinline,optnone)) forcebt2(int level) {
  volatile int level_2 = level * 2;
  DoNotOptimize(level_2);
  level_2 = forcebt3(level_2);
  return level_2;
}
static int __attribute__((noinline,optnone)) forcebt1(int level) {
  volatile int level_1 = level * 2;
  DoNotOptimize(level_1);
  level_1 = forcebt2(level_1);
  return level_1;
}

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  volatile int level_0 = 2;
  DoNotOptimize(level_0);
  level_0 = forcebt1(level_0);
  return 0;
}
#pragma GCC pop_options
