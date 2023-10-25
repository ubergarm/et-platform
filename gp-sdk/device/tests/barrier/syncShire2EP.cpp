
/*-------------------------------------------------------------------------
 * Copyright (C) 2022, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h> 

#include "entryPoint.h"

#include "sync.h"
#include "CommonCode.h"

#include "barrierKernelArguments.h"

static constexpr int maxBarrierSize = 64;
static constexpr int numTests = 6; // number of barrier sizes to test
// const size_t barrierSize[numTests] = {64,4,8,16,32,maxBarrierSize};
const size_t barrierSize[numTests] = {2,4,8,16,32,maxBarrierSize};

size_t result[2048] = {0};

int entryPoint_0(KernelArguments* args);
auto getFirstThreadinSyncGroup(int threadId, size_t bsize) -> int;
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, entryPoint_0);

// Obtain the first thread in the group
auto getFirstThreadinSyncGroup(int threadId, size_t bsize) -> int {
    const auto countLog2 = 64UL - 1UL - __builtin_clzll(bsize);
    auto barrierStart = (threadId >> countLog2) << countLog2;
    return barrierStart;
}

__attribute__((noinline)) int entryPoint_0([[maybe_unused]] KernelArguments*  args) {
  static constexpr uint64_t one = 1;
  static constexpr size_t numAdds = 1;

  // This is a test for synchonization on each shire.
  // The n-th thread performs n*512 additions in the n-th position of the vector
  auto threadId = get_relative_thread_id();
  auto numThreads = get_num_threads();

  // exit if not: 0 <= threadId < numThreads
  if (threadId >= numThreads || threadId < 0 ) {
     return 0;
  }
  // Kernel arguments
  uint64_t preValue = 0;
  for (int c = 0; c < numTests; c++) {
    auto bsize = barrierSize[c];

    if (threadId == 0) {
      et_printf("Testing sync group size = %lu\n", bsize);
    }

    auto syncGroupMasterThread = getFirstThreadinSyncGroup(threadId, bsize);

    // Last thread of each group will perform additions
    auto isLastThreadInSyncGroup = ((threadId + 1) % bsize) == 0;
    // et_printf("[T:%d M:%s] Testing barrier(start=%d, count=%lu)\n", threadId, isLastThreadInSyncGroup ? "true" : "false", syncGroupMasterThread, bsize);
    if (isLastThreadInSyncGroup) {
      
      // set result[syncGroupMasterThread] = 0;
      uint64_t tmp;
      __asm__ volatile("amoorg.d %[tmp], x0, (%[resultPtr])\n"                                      
              : [tmp]  "=r" (tmp)      
              : [resultPtr]  "r" (&result[syncGroupMasterThread])
      );
      // et_printf("[BS:%lu][T:%d] startvalue = %lu\n", bsize, threadId, tmp);

      // Add until result[syncGroupMasterThread] = numAdds;
      for (size_t i = 0; i < numAdds; i++) {
        // AMOADDG.D global atomic add
        __asm__ __volatile__("amoaddg.d %[preValue], %[one], (%[dataPtr])\n"
                            : [ preValue ] "=r" (preValue)
                            : [ dataPtr ] "r" (&result[syncGroupMasterThread]),
                              [ one ] "r" (one)
                            :);
      }
    }
    // Synchronize each group. There will be num groups = numThreads / bsize
    hart::barrier(syncGroupMasterThread, bsize);

    // Check the results are correct
    if (threadId == syncGroupMasterThread) {
      // Global load result
      uint64_t resultValue;
      __asm__ volatile("amoorg.d %[resultValue], x0, (%[resultPtr])\n"                                      
              : [resultValue]  "=r" (resultValue)      
              : [resultPtr]  "r" (&result[syncGroupMasterThread])
      );

      // first thread of each group checks the result = numAdds
      if (numAdds != resultValue) {
        et_printf("Invalid value: result[%d]=%lu (should be: %lu)\n", syncGroupMasterThread, resultValue, numAdds);
        et_printf("Address: %p\n", &result[syncGroupMasterThread]);
        return -1;
      }
      // et_printf("Correct result[%d]=%lu \n", barrierStart, resultValue);
      // reset result vector to 0
      __asm__ __volatile__("amoandg.d %[preValue], x0, (%[dataPtr])\n"
                            : [ preValue ] "=r" (preValue)
                            : [ dataPtr ] "r" (&result[syncGroupMasterThread])
                            :);                    
    }
    
    // Sync all threads before moving into the next count value
    hart::barrier();
  }
  if (threadId == 0)
    et_printf("Success\n");

  return 0;
}

