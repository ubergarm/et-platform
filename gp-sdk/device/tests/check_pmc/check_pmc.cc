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

#include "entryPoint.h"
#include <cstdlib>
#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>

#include "profiling.h"
#include "environment.h"


int entryPoint_0(KernelArguments* args);
int entryPoint_1(KernelArguments* args);
extern  DeviceConfig config {2, entryPoint_0, entryPoint_1};

static inline int checkPMC() {

  uint64_t counters[64] = {0};
  auto fail = false;

  auto start = PMC_Get_Current_Cycles();

  for (auto i = 0;; i += 3) {
    // hammer cycles read mixing with some other PMC
    auto val = PMC_Get_Current_Cycles();
    volatile uint64_t other = pmu_core_counter_read_unpriv(HPM_COUNTER_4);
    auto val2 = PMC_Get_Current_Cycles();
    auto val3 = PMC_Get_Current_Cycles();
    counters[i % 64] = val;
    counters[(i + 1) % 64] = val2;
    counters[(i + 2) % 64] = val3;
    (void)other;

    // check counter consistency
    auto d1 = std::abs(int64_t(val3) - int64_t(val2));
    auto d2 = std::abs(int64_t(val2) - int64_t(val));
    auto d3 = std::abs(int64_t(val3) - int64_t(val));
    auto valPrev = (i > 0) ? counters[(i - 1) % 64] : 0;
    auto d4 = (i > 0) ? std::abs(int64_t(val) - int64_t(valPrev)) : 0;

    constexpr auto thr = 1000000;
    auto f1 = d1 > thr;
    auto f2 = d2 > thr;
    auto f3 = d3 > thr;
    auto f4 = d4 > thr;

    if (f1 or f2 or f3 or f4) {
      et_printf("%d %d -- fail  -- [ %lu %lu %lul %lu ] \n", get_hart_id(), i, val, val2, val3, valPrev);
      fail = true;
      break;
    }

    // declare a pass after 10 secs (@600Mhz).
    constexpr auto timeLImit = 10ul * 600000000ul;
    if ((start < val3) and (val3 - start) > timeLImit) {
      et_printf("%d test complete %lu %lu  %lu %lu", get_hart_id(), val3, start, val3 - start, timeLImit);
      break;
    }
  }

  // dump the last 64 counters read.
  for (auto i = 0; i < 64; i++) {
    et_printf("%d -- [ %lu ] \n", get_hart_id(), counters[i]);
  }

  et_assert(!fail && "test failed");

  return 0;
}

// both threads on the minion do the same.
int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  return checkPMC();
}

int entryPoint_1([[maybe_unused]] KernelArguments* args) {
  return checkPMC();
}
