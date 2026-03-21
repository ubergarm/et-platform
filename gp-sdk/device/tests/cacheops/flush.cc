/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#include "entryPoint.h"
#include <cstdlib>

#include "sync.h"
#include <etsoc/common/utils.h>
#include <etsoc/isa/cacheops-umode.h>
#include <etsoc/isa/hart.h>

class KernelArguments;
int entryPoint(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint, nullptr);

struct alignas(64) TestElement {
  uint64_t element = 0;
};
alignas(64) static TestElement buffer[2048] = {0};

int entryPoint([[maybe_unused]] KernelArguments* args) {
  auto threadId = get_relative_thread_id();
  // produce 1 value for this thread
  buffer[threadId].element = (0x1234ull << 16) | threadId;

  /* Flush up to L3
   * This function flushes the specified virtual address from the cache hierarchy,
   * if it is present and dirty, up to the provided cache level. Optionally,
   * a repeat count can be specified to flush more lines, whose addresses are
   * calculated sing the provided stride. Optionally,
   * each potential line flush can be gated by the value of the TensorMask CSR.
   */
  constexpr uint64_t useTensorMask = 0;
  constexpr uint64_t numLinesMinusOne = 1 - 1;

  // Flushing cache op requires a fence before and a tensor wait after
  __asm__ volatile("fence\n" ::: "memory");
  cache_ops_flush_va(useTensorMask, cop_dest::to_L3, (uint64_t)&buffer[threadId], numLinesMinusOne, 0, 0);
  __asm__ __volatile__("csrwi tensor_wait, 6\n" : :);
  
  hart::barrier(); // All assigned threads synchronize

  // each thread reads data produced by prev ones... let's skip 2 for the sake of simplicity.

  if (threadId >= 2) { 
    auto producerId = threadId - 2;
    auto val = buffer[producerId].element;
    et_assert(val == ((0x1234ull << 16) | producerId));
  }

  return 0;
}
