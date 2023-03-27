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

#include <bitset>
#include <stdio.h>

#include "CommonCode.h"
#include "entryPoint.h"
#include <system/abi.h>
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

/* TLS storage linker script labels */
extern uint8_t* __tdata_start;
extern uint8_t* __tbss_end;
extern uint8_t* __tls_alloc_start;

/* Linker label used to rebase entryPoint function pointers */
extern const uint32_t _text_init_start;
// Generic C function pointer.
using function_t = void (*)();
/* Linker labels for initialization and fini arrays */
extern const function_t __preinit_array_start;
extern const function_t __preinit_array_end;
extern const function_t __init_array_start;
extern const function_t __init_array_end;
extern const function_t __fini_array_start;
extern const function_t __fini_array_end;

// Kernel configuration object. needs to be declared by the user throuch DECLARE_DEVICE_CONFIG
namespace device_config {
extern DeviceConfig config;
/* Global pointer to environment struct allocated by the host */
const kernel_environment_t * env_;
const kernel_environment_t fallback_env = {{0,0,0,0}, 0xFFFFFFFF, 600};
}

/* Number of times the kernel has been launched */
uint64_t numberOfBoots __attribute__((section("persistentData"))) = {1};

void resetBSS();
void resetData();
bool hasGlobalData();
extern "C" int deviceGpSdkEntry(void* args, kernel_environment_t* env);

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

/// returns true if  there are functions to call on init_arrays
static bool hasInitArrays() {
  return !(__preinit_array_start == __preinit_array_end) && !(__init_array_start == __init_array_end);
}

/// Initialize per hart  Thread Local Storage.
/// note: each hart in the system should call this function.
static void initializeTLS() {

  auto tlsSize = (&__tbss_end - &__tdata_start) * sizeof(__tbss_end);
  if (tlsSize == 0) {
    return;
  }
  auto tlsStart = &__tdata_start;
  auto hartId = get_hart_id();
  auto tlsHartBase = &__tls_alloc_start + (hartId * tlsSize);

  // populate tls section for this hart.
  local_memcpy(tlsHartBase, tlsStart, tlsSize);

  // initialize tp with the tls Base address for this hart
  asm volatile("mv tp, %[tlsHartBase] \n" : : [ tlsHartBase ] "r"(tlsHartBase));
}

/// @brief Computes the new memory address of a function after loading the kernel in the device.
/// @param fnc function pointer which address has been determined in compile time.
/// @param base new base address of the program
/// @return Returns the new rebased pointer to the function
template <typename T> auto rebaseFunction(T fnc, uint64_t base) -> decltype(fnc) {
  return (decltype(fnc))((uint64_t)fnc + base);
}

/// calls init_array functions
static void callInitArrayFunctions() {

  for (const function_t* entry = &__preinit_array_start; entry < &__preinit_array_end; ++entry) {
    auto func = rebaseFunction(*entry, (uint64_t)&_text_init_start);
    (*func)();
  }

  for (const function_t* entry = &__init_array_start; entry < &__init_array_end; ++entry) {
    auto func = rebaseFunction(*entry, (uint64_t)&_text_init_start);
    (*func)();
  }
}

// FIXME: setting args inhibits fast-path initialization
static bool needToSetArgs() {
  return true;
}

extern "C" int deviceGpSdkEntry(void* args, kernel_environment_t* env) {
  uint32_t hart = get_hart_id();
  uint32_t threadId = hart & 1;
  uint32_t minionId = hart >> 1;
  uint32_t shireId = minionId >> 5;
  minionId = minionId & 0x1F;
  uint64_t shireMask = (env != nullptr)? env->shire_mask : 0xFFFFFFFF;

  auto needSync = hasGlobalData() || hasInitArrays() || needToSetArgs();

  if (shireId >= 32)
    return 0;

  if (device_config::config.threadsPerCore == 1 && threadId == 1) {
    // exit if thread 1 is unused
    return 0;
  }

  // fast-path: no global data: fast-forward to user code.
  if (!needSync) {
    if (device_config::config.threadsPerCore == 1) {
      initializeTLS();
      KernelEntryPointFuncPtr rebasedFnc =
        rebaseFunction(device_config::config.entryPoint_0, (uint64_t)(&_text_init_start));
      return rebasedFnc(args);
    } else {
      if (threadId == 0) {
        initializeTLS();
        KernelEntryPointFuncPtr rebasedFnc =
          rebaseFunction(device_config::config.entryPoint_0, (uint64_t)(&_text_init_start));
        return rebasedFnc(args);
      } else {
        initializeTLS();
        KernelEntryPointFuncPtr rebasedFnc =
          rebaseFunction(device_config::config.entryPoint_1, (uint64_t)(&_text_init_start));
        return rebasedFnc(args);
      }
    }
  }

  // global data: initialize and forwared to user-code.
 if (get_relative_thread_id(shireMask) == 0) {
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

    // call init array (dynamic initializatoin) functions from thrad 0
    callInitArrayFunctions();
    if (env != nullptr) {
      device_config::env_ = env;
    } else {
      device_config::env_ = &device_config::fallback_env;
    }
    evictCacheLine(cop_dest::to_L3, (uint8_t*)&device_config::env_);
    wakeUpThreads(shireMask);
  }

  // Wait initialization to complete and forward to user-code.
  if (device_config::config.threadsPerCore == 1) {
    if (threadId == 0) {
      fcc_consume(FCC_0);
      initializeTLS();
      KernelEntryPointFuncPtr rebasedFnc =
        rebaseFunction(device_config::config.entryPoint_0, (uint64_t)(&_text_init_start));
      return rebasedFnc(args);
    }
  } else {
    if (threadId == 0) {
      fcc_consume(FCC_0);
      initializeTLS();
      KernelEntryPointFuncPtr rebasedFnc =
        rebaseFunction(device_config::config.entryPoint_0, (uint64_t)(&_text_init_start));
      return rebasedFnc(args);
    } else {
      fcc_consume(FCC_0);
      initializeTLS();
      KernelEntryPointFuncPtr rebasedFnc =
        rebaseFunction(device_config::config.entryPoint_1, (uint64_t)(&_text_init_start));
      return rebasedFnc(args);
    }
  }
  return 0;
}

int get_num_threads() {
  return __builtin_popcountll(device_config::env_->shire_mask) * SOC_MINIONS_PER_SHIRE *  device_config::config.threadsPerCore;;
}

int get_relative_thread_id() {
  return get_relative_thread_id(device_config::env_->shire_mask);
}

inline int get_relative_thread_id(uint64_t shireMask) {
  constexpr int NUM_HARTS_PER_MINION = 2;
  auto hartId = static_cast<int>(get_hart_id());
  int startingHart = static_cast<int>(__builtin_ctzll(shireMask) * SOC_MINIONS_PER_SHIRE * 2);

  // return -1 ifs not an active thread
  if (hartId < startingHart) {
    return -1;
  }

  int threadId = (hartId / (NUM_HARTS_PER_MINION / device_config::config.threadsPerCore)) - startingHart;
  return threadId;
}
