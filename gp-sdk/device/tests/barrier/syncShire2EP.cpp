
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

constexpr uint64_t one = 1;

constexpr int vsize = 6;
constexpr int maxThread = 64;
const size_t countValues[vsize] = {2,4,8,16,32,maxThread};
const size_t numAdds = 1;
size_t result[maxThread] = {0};

int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, entryPoint_0);

__attribute__((noinline)) int entryPoint_0([[maybe_unused]] KernelArguments*  args) {
  // This is a test for synchonization on each shire.
  // The n-th thread performs n*512 additions in the n-th position of the vector
  auto threadId = get_relative_thread_id();
  auto numThreads = get_num_threads();

  // // only first 64 threads
  // if (threadId >= maxThread)
  //   return 0;

  if (threadId >= numThreads || threadId < 0 ) {
     return 0;
  }
  // Kernel arguments
  uint64_t preValue = 0;

  for (int c = 0; c < vsize; c++) {
    if (threadId < maxThread) {
      auto count = countValues[c];
      auto isCountMultiple = ((threadId + 1) % count) == 0;

      if (threadId == 0) {
        et_printf("Testing group size = %lu\n", count);
      }

      // Obtain the first thread in the group
      // log2(x) = (sizeof(x) * 8 - 1) - count_leading_0s(x);
      auto countLog2 = 64UL - 1UL - __builtin_clzll(count);
      auto barrierStart = (threadId >> countLog2) << countLog2;

      // et_printf("[T:%d M:%s] Testing barrier(start=%d, count=%lu)\n", threadId, isCountMultiple ? "true" : "false", barrierStart, count);

      // Last thread of each group will perform additions
      if (isCountMultiple) { 

        // testing
        uint64_t tmp;
        __asm__ volatile("amoorg.d %[tmp], x0, (%[resultPtr])\n"                                      
                : [tmp]  "=r" (tmp)      
                : [resultPtr]  "r" (&result[barrierStart])
        );
        // et_printf("Start adding [GS:%lu][T:%d] startvalue = %lu\n", count, threadId, tmp);

        for (size_t i = 0; i < numAdds; i++) {
          // AMOADDG.D global atomic add
          __asm__ __volatile__("amoaddg.d %[preValue], %[one], (%[dataPtr])\n"
                              : [ preValue ] "=r" (preValue)
                              : [ dataPtr ] "r" (&result[barrierStart]),
                                [ one ] "r" (one)
                              :);
        }
      }
      hart::barrier(barrierStart, count);

      if (threadId == barrierStart) {
        // Global load result
        uint64_t resultValue;
        __asm__ volatile("amoorg.d %[resultValue], x0, (%[resultPtr])\n"                                      
                : [resultValue]  "=r" (resultValue)      
                : [resultPtr]  "r" (&result[barrierStart])
        );

        // first thread of each group checks the result = numAdds
        if (numAdds != resultValue) {
          et_printf("Invalid value: result[%d]=%lu (should be: %lu)\n", barrierStart, resultValue, numAdds);
          et_printf("Address: %p\n", &result[barrierStart]);
          return -1;
        }
          // et_printf("Correct result[%d]=%lu \n", barrierStart, resultValue);
        // reset result vector to 0
        __asm__ __volatile__("amoandg.d %[preValue], x0, (%[dataPtr])\n"
                              : [ preValue ] "=r" (preValue)
                              : [ dataPtr ] "r" (&result[barrierStart])
                              :);                    
      }
    }
    // Sync all threads before moving into the next count value
    hart::barrier();
  }
  if (threadId == 0)
    et_printf("Success\n");

  return 0;
}

