
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

int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(1, entryPoint_0, nullptr);


__attribute__((noinline)) int entryPoint_0(KernelArguments* args) {
  // This test creates a symmetric vector using atomic additions.
  // The n-th thread performs n +1 additions in the n-th position of the vector
  // In a second step, n-th thread performs data[minionId] += data[numMinions - minionId - 1];
  // The result is a vector where all values in vector are numMinions-1
  // i.e.: data[0, 1, 2 ... n-2, n-1, n] = { 1023, 1023 ... 1023 }

  // Because there is unbalance between threads, any non-sync thread would produce incorrect results.
  // Result correctness is easy to check.

  auto threadId = get_relative_thread_id();
  auto numThreads = get_num_threads();

  if (threadId >= numThreads || threadId < 0 ) {
     return 0;
  }
  // Kernel arguments
  auto data = args->data; 
  auto accumData = args->accumData;
  uint64_t preValue;

  // unbalanced number of operations per minion
  for (int i = 0; i < threadId; i++) {
    auto dataPtr = &data[threadId];

    // AMOADDG.D global atomic add
    __asm__ __volatile__("amoaddg.d %[preValue], %[one], (%[dataPtr])\n"
                         : [ preValue ] "=r" (preValue)
                         : [ dataPtr ] "r" (dataPtr),
                           [ one ] "r" (one)
                         :);
  }

  // copy data[] to accumData
  uint64_t srcValue;
  auto srcPtr = &data[threadId];
  auto dstPtr = &accumData[threadId];

  // LOAD 
  __asm__ volatile("amoorg.d %[srcValue], x0, (%[srcPtr])\n"                                      
                  : [srcValue]  "=r" (srcValue)      
                  : [srcPtr]  "r" (srcPtr)
  );

  // STORE
  __asm__ __volatile__("amoaddg.d %[preValue], %[srcValue], (%[dstPtr])\n"
                      : [ preValue ] "=r" (preValue)
                      : [ dstPtr ] "r" (dstPtr),
                        [ srcValue ] "r" (srcValue)
                      :);

  hart::barrier();

  srcPtr = &data[numThreads - threadId - 1];
  dstPtr = &accumData[threadId];

  // Atomic global load (using an OR), srcValue = data[assignedMinions - threadId - 1];
  __asm__ volatile("amoorg.d %[srcValue], x0, (%[srcPtr])\n"                                      
                  : [srcValue]  "=r" (srcValue)
                  : [srcPtr]  "r" (srcPtr)
  );

  // data[minionId] += data[assignedMinions - threadId - 1];
  __asm__ __volatile__("amoaddg.d %[preValue], %[srcValue], (%[dstPtr])\n"
                      : [ preValue ] "=r" (preValue)
                      : [ dstPtr ] "r" (dstPtr),
                        [ srcValue ] "r" (srcValue)
                      :);

  hart::barrier();

  // All values in the vector should add the same
  if (threadId == 0) {
    auto checkValue = accumData[threadId];

    for (int i = 1; i < numThreads; i++) {
      if (checkValue != accumData[i]) {
        et_printf("Invalid value data[%d] %lu\n", i, accumData[i]);
        return -1;
      }
    }
    et_printf("Success\n");
  }
  return 0;
}
