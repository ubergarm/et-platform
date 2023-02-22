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

#include <stdio.h>
#include <bitset>

#include "CommonCode.h"
#include "entryPoint.h"
#include "environment.h"
#include <etsoc/common/utils.h>
#include <etsoc/isa/barriers.h>
#include <etsoc/isa/cacheops-umode.h>
#include <etsoc/isa/hart.h>

/* Linker labels to global .bss and .data sections */
extern uint32_t _bss_end;
extern uint32_t _bss_start;
extern uint32_t _data_end;
extern uint32_t _data_start;
extern uint32_t _data_ro_copy_end;
extern uint32_t _data_ro_copy_start;
/* Linker label used to rebase entryPoint function pointers */
extern uint32_t _text_init_start;

extern DeviceConfig config;

/* Global pointer to arguments provided by the host */
Arguments * args_;

/* Number of times the kernel has been launched */
uint64_t numberOfBoots __attribute__((section("persistentData"))) = {1};

void resetBSS();
void resetData();
bool hasGlobalData();
extern "C" int deviceGpSdkEntry(KernelArguments* args);

/* wake up all threads on the shire-Mask group system */
static inline void wakeUpThreads(uint64_t shire_mask) {

  /* uses ESR-broadcast extension to send credits to all shires simultaneously */
  volatile uint64_t* broadcast_data =
    (volatile uint64_t*)ESR_SHIRE_PROT_ADDR(PRV_U, THIS_SHIRE, ESR_SHIRE_BROADCAST0); // 0x013ff5fff0
  volatile uint64_t* broadcast_address =
    (volatile uint64_t*)ESR_SHIRE_PROT_ADDR(PRV_U, THIS_SHIRE, ESR_SHIRE_BROADCAST1); // 0x013ff5fff8

  constexpr uint64_t thread_mask = 0xffffffff;
  *broadcast_data = thread_mask;
  // broadcast to threads 0
  *broadcast_address =
    (shire_mask & 0xFFFFFFFF) | (((ESR_SHIRE(0, FCC_CREDINC_0) >> 3) & 0x7FFFF) << ESR_BROADCAST_ESR_ADDR_SHIFT);
  // braodcast to threads 1.
  *broadcast_address =
    (shire_mask & 0xFFFFFFFF) | (((ESR_SHIRE(0, FCC_CREDINC_2) >> 3) & 0x7FFFF) << ESR_BROADCAST_ESR_ADDR_SHIFT);
}

/// @brief Resets global memory region .bss to zero
void resetBSS() {
  uint8_t* bss_end = (uint8_t*)&_bss_end;
  uint8_t* bss_start = (uint8_t*)&_bss_start;
  global_memset(bss_start, 0, bss_end - bss_start);
}

/// @brief Initializes global memory region .data to its original value
void resetData() {
  uint8_t* data_end = (uint8_t*)&_data_end;
  uint8_t* data_start = (uint8_t*)&_data_start;
  uint8_t* data_ro_copy_start = (uint8_t*)&_data_ro_copy_start;

  if (numberOfBoots == 1) {
    // backup .data section
    global_memcpy(data_ro_copy_start, data_start, data_end - data_start);
  } else {
    // restore .data section
    global_memcpy(data_start, data_ro_copy_start, data_end - data_start);
  }
}

/// @brief returns true if Global Data sections (.bss and .data) are not empty.
bool hasGlobalData() {
  uint8_t* bss_end = (uint8_t*)&_bss_end;
  uint8_t* bss_start = (uint8_t*)&_bss_start;
  uint8_t* data_end = (uint8_t*)&_data_end;
  uint8_t* data_start = (uint8_t*)&_data_start;
  return !((bss_end == bss_start) && (data_end == data_start));
}

/// @brief Computes the new memory address of a function after loading the kernel in the device.
/// @param fnc function pointer which address has been determined in compile time.
/// @param base new base address of the program
/// @return Returns the new rebased pointer to the function
KernelEntryPointFuncPtr rebaseEntryPointPtr(KernelEntryPointFuncPtr fnc, uint64_t base) {
  return (KernelEntryPointFuncPtr)((uint64_t) fnc + base);
}

extern "C" int deviceGpSdkEntry(KernelArguments* args) {
  uint32_t hart = get_hart_id();
  uint32_t threadId = hart & 1;
  uint32_t minionId = hart >> 1;
  uint32_t shireId = minionId >> 5;
  minionId = minionId & 0x1F;
  uint32_t globalMinionId = shireId * 32 + minionId;
  auto needSync = hasGlobalData();
  
  if (shireId >= 32)
    return 0;

  // fast-path: no global data: fast-forward to user code.
  if (!needSync) {
    if (config.threadsPerCore == 1) {
      if(threadId == 0) {
        KernelEntryPointFuncPtr rebasedFnc = rebaseEntryPointPtr(config.entryPoint_0,(uint64_t)(&_text_init_start));
        return rebasedFnc(args);
      } else {
        return 0;
      }
    } else {
      if (threadId == 0) {
        KernelEntryPointFuncPtr rebasedFnc = rebaseEntryPointPtr(config.entryPoint_0,(uint64_t)(&_text_init_start));
        return rebasedFnc(args);
      } else {
        KernelEntryPointFuncPtr rebasedFnc = rebaseEntryPointPtr(config.entryPoint_1,(uint64_t)(&_text_init_start));
        return rebasedFnc(args);
      }
    }
  }

  
  // global data: initialize and forwared to user-code.
  if (globalMinionId == 0 && threadId == 0) {
    // Reset .bss and .data sections on each kernel launch
    resetBSS();
    resetData();

    // increase number of boots atomically
    const uint32_t increment = 1;
    uint32_t result;
    __asm__ __volatile__("amoaddg.w %[result], %[increase], (%[dst])\n"
                         : [ result ] "=r"(result)
                         : [ increase ] "r"(increment), [ dst ] "r"(&numberOfBoots)
                         :);

    args_ = (Arguments *) args;
    args_->env.numThreads = __builtin_popcountll(args_->env.shireMask) * SOC_MINIONS_PER_SHIRE * config.threadsPerCore;
    evictCacheLine(0x3, (uint8_t *) &args_);
    evictCacheLine(0x3, (uint8_t *) args_);
    et_printf("th: %d\n", args_->env.numThreads);
    constexpr uint64_t full_shire_mask = 0xffffffff;
    wakeUpThreads(full_shire_mask);
  }

  // Wait initialization to complete and forward to user-code.
  if (config.threadsPerCore == 1) {
    if(threadId == 0) {
      fcc_consume(FCC_0);
      KernelEntryPointFuncPtr rebasedFnc = rebaseEntryPointPtr(config.entryPoint_0,(uint64_t)(&_text_init_start));
      return rebasedFnc(args);
    }
  } else {
    if (threadId == 0) {
      fcc_consume(FCC_0);
      KernelEntryPointFuncPtr rebasedFnc = rebaseEntryPointPtr(config.entryPoint_0,(uint64_t)(&_text_init_start));
      return rebasedFnc(args);
    } else {
      fcc_consume(FCC_0);
      KernelEntryPointFuncPtr rebasedFnc = rebaseEntryPointPtr(config.entryPoint_1,(uint64_t)(&_text_init_start));
      return rebasedFnc(args);
    }
  }
  return 0;
}

int get_num_threads() {
  return args_->env.numThreads;
}

int get_relative_thread_id() {
  constexpr int maxThreadsPerCore = 2;
  auto hartId = static_cast<int>(get_hart_id());
  int startingHart = static_cast<int>(__builtin_ctzll(args_->env.shireMask) * SOC_MINIONS_PER_SHIRE * 2);

  // return -1 ifs not an active thread
  if (hartId < startingHart) {
    return -1;
  }

  int threadId = (hartId / (maxThreadsPerCore / config.threadsPerCore)) - startingHart;
  return threadId;
}