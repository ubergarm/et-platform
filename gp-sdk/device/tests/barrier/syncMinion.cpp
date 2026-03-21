
/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
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

int entryPoint_0(KernelArguments* args);
int entryPoint_1(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, entryPoint_1);

constexpr uint64_t one = 1;

__attribute__((noinline)) int entryPoint_0(KernelArguments* args) {
  // This test creates a symmetric vector using atomic additions.
  // The n-th thread performs n +1 additions in the n-th position of the vector
  // In a second step, n-th thread performs data[threadId] += data[numMinions - threadId - 1];
  // The result is a vector where all values in vector are numMinions-1
  // i.e.: data[0, 1, 2 ... n-2, n-1, n] = { 1023, 1023 ... 1023 }

  // Because there is unbalance between threads, any non-sync thread would produce incorrect results.
  // Result correctness is easy to check. 
  auto threadId = get_relative_thread_id();
  auto numThreads = get_num_threads();
  auto elementId = threadId >> 1;
  auto numElements = numThreads >> 1;


  if (threadId >= numThreads) {
      return 0;
  }

  // Kernel arguments
  auto data = args->data; 
  auto accumData = args->accumData;
  uint64_t preValue;

  // unbalanced number of operations per minion
  for (int i = 0; i < elementId; i++) {
    auto dataPtr = &data[elementId];

    // AMOADDG.D global atomic add
    __asm__ __volatile__("amoaddg.d %[preValue], %[one], (%[dataPtr])\n"
                         : [ preValue ] "=r" (preValue)
                         : [ dataPtr ] "r" (dataPtr),
                           [ one ] "r" (one)
                         :);
  }


  // copy data[] to accumData
  uint64_t srcValue;
  auto srcPtr = &data[elementId];
  auto dstPtr = &accumData[elementId];

  // LOAD 
  __asm__ volatile("amoorg.d %[srcValue], x0, (%[srcPtr])\n"                                      
                  : [srcValue]  "=r" (srcValue)      
                  : [srcPtr]  "r" (srcPtr)
  );

  // et_printf("elementId = %d, threadId = %d, value = %lu\n", elementId,  threadId, srcValue);

  // STORE
  __asm__ __volatile__("amoaddg.d %[preValue], %[srcValue], (%[dstPtr])\n"
                      : [ preValue ] "=r" (preValue)
                      : [ dstPtr ] "r" (dstPtr),
                        [ srcValue ] "r" (srcValue)
                      :);

  hart::barrier();
  hart::barrier<hart::Scope::minion>();
  // Threads 1 will start computing

  // Wait for Threads 1 to finish
  hart::barrier<hart::Scope::minion>();
  hart::barrier();

  // All values in the vector should add the same
  if (threadId == 0) {
    auto checkValue = accumData[0];
    for (int i = 1; i < numElements; i++) {
      if (checkValue != accumData[i]) {
        et_printf("Invalid value data[%d] %lu\n", i, accumData[i]);
        return -1;
      }
    }
    et_printf("Success\n");
  }
  return 0;
}

__attribute__((noinline)) int entryPoint_1(KernelArguments* args) {

 auto threadId = get_relative_thread_id();
 auto numThreads = get_num_threads();
 auto elementId = threadId >> 1;
 auto numElements = numThreads >> 1;


 if (threadId >= numThreads) {
     return 0;
 }

  
  // thread_1 wont start until thread 0 has finished.
  hart::barrier<hart::Scope::minion>();

  // Kernel arguments
  auto data = args->data;
  auto accumData = args->accumData;
  uint64_t preValue;

  // copy data[] to accumData
  uint64_t srcValue;
  auto srcPtr = &data[numElements - elementId - 1];
  auto dstPtr = &accumData[elementId];

  // Atomic global load (using an OR), srcValue = data[assignedMinions - threadId - 1];
  __asm__ volatile("amoorg.d %[srcValue], x0, (%[srcPtr])\n"                                      
                  : [srcValue]  "=r" (srcValue)      
                  : [srcPtr]  "r" (srcPtr)
  );

  // et_printf("elementId = %d, threadId = %d, value = %lu\n", elementId,  threadId, srcValue);

  // data[threadId] += data[assignedMinions - threadId - 1];
  __asm__ __volatile__("amoaddg.d %[preValue], %[srcValue], (%[dstPtr])\n"
                      : [ preValue ] "=r" (preValue)
                      : [ dstPtr ] "r" (dstPtr),
                        [ srcValue ] "r" (srcValue)
                      :);

  hart::barrier<hart::Scope::minion>();

  return 0;

}

