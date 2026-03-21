/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>
#include <algorithm>

#include "sync.h"
#include "CommonCode.h"
#include "entryPoint.h"
#include "sdot_kernel_arguments.h"
#include <etsoc/isa/tensors.h>

int entryPoint_0(KernelArguments* args);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);


//clang-format off
static inline __attribute__((always_inline))
void sdot_vector(const size_t begin, const size_t end,
  const float* const x,  const float* const y, float* const w)
{
    if(begin >= end){
      return;
    }
    
    constexpr int vlen = 8;
    constexpr uint32_t mask = 0xff;
#ifndef __clang__
    mask_set(0, mask);
#endif
    auto i = begin;
    auto loop_end =  end > (vlen - 1) ? end - (vlen - 1) : (size_t)0;
    for (; i < loop_end; i += vlen) {
        float xValue;
        float yValue;
        const float *xv = &x[i]; 
        const float *yv = &y[i];
        const float *wv = &w[i];
        // load x and y
        __asm__ __volatile__ ("flw.ps %[xValue], 0(%[xv])\n"
                              "flw.ps %[yValue], 0(%[yv])\n"
                              : [xValue] "=&f"(xValue),
                                [yValue] "=&f"(yValue)
                              : [xv] "r" (xv), [yv] "r" (yv)

                          #ifdef __clang__
                              , [ mask ] "M"(mask) 
                          #endif
                              : );

        // w[i] = x[i] * y[i] 
        __asm__ __volatile__(
                            // multipy datavalues
                            "fmul.ps %[yValue], %[xValue], %[yValue]\n"
                            : [yValue] "+&f"(yValue)
                            : [xValue] "f"(xValue)
                          #ifdef __clang__
                            , [ mask ] "M"(mask) 
                          #endif
                            :);

        // store w[i]
        __asm__ __volatile__ ("fsw.ps %[yValue], 0(%[wv])\n"
                              :         
                              : [wv] "r" (wv), [yValue] "f"(yValue)
                          #ifdef __clang__
                              , [ mask ] "M"(mask)
                          #endif 
                              : );

      for (size_t j = i+1; j < i + vlen; j++){
        w[i] += w[j];
      }
    }

    if (i < end){
      w[i] = x[i] * y[i];
      for (size_t j = i+1; j < end; ++j) {
        w[i] += x[j] * y[j];
      }
    }

    // sum all calculated values and store in wv[begin]
    for(size_t j = begin + vlen; j < end; j+=vlen){
      w[begin] += w[j];
    }
 
    evictCacheLine(0x1ULL, (uint8_t*)&y[begin]);
}
//clang-format on

int entryPoint_0(KernelArguments* vectors) {
  auto minionId = get_relative_thread_id();
  
  size_t numWorkers = SOC_MINIONS_PER_SHIRE; // just 1 shire (32 minions).
  size_t elemsPerWorker = (vectors->numElements + numWorkers - 1) / numWorkers;
  if (elemsPerWorker % 16) {
    elemsPerWorker += 16 - (elemsPerWorker % 16);
  }
  size_t begin = elemsPerWorker * minionId;
  size_t end = std::min(elemsPerWorker * (minionId + 1), vectors->numElements);
#ifdef SDOT_VECTOR
  sdot_vector(begin, end, vectors->x, vectors->y, vectors->y);
#else
  float localSum = 0;
  if(begin < end){
    for (size_t i = begin; i < end; ++i) {
      localSum += vectors->x[i] * vectors->y[i];
    }
    // write to y[begin] to implicity avoid false sharing
    vectors->y[begin] = localSum;
    evictCacheLine(0x1ULL, (uint8_t*)&vectors->y[begin]);
  }
#endif
  hart::barrier();

  if(minionId == 0){
    float result = 0;
    for(size_t i = 0; i < vectors->numElements; i+=elemsPerWorker){
      result += vectors->y[i];
    }
    *(vectors->res) = result;
  }
  return 0;
}
