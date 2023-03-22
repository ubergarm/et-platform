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

#ifndef _ENTRY_POINT_H_
#define _ENTRY_POINT_H_

#include "environment.h"

/* EntryPoint kernel function prototype
 * note that user entry points can receive typed ptr (namely KernelArguments *)*/
using KernelEntryPointFuncPtr = int (*)(void*);

/// @brief __DeviceConfig

struct alignas(64) __DeviceConfig {
  int32_t threadsPerCore = 0;
  KernelEntryPointFuncPtr entryPoint_0 = nullptr;
  KernelEntryPointFuncPtr entryPoint_1 = nullptr;
};

// @brief DeviceConfig: type alias to enforce const definition.
using DeviceConfig = const __DeviceConfig;

/// @brief macro for registering a single DeviceConfig singleton
/// \param threadsPerCore threads per minion used
/// \param entry0 entryPoint for ET-Minion thread 0 (Note, If noly 1 entry point is required it should be 0)
/// \param entry1 entryPoint for ET-Minion thread 1
//
// Examples:
// /// Examples of DeviceConfig
/// DECLARE_KERNEL_ENTRY_POINTS(1,entryPoint_0, nullptr);
/// DECLARE_KERNEL_ENTRY_POINTS(2,entryPoint_0, entryPoint_1);
/// DECLARE_KERNEL_ENTRY_POINTS(2,entryPoint, entryPoint);
#define DECLARE_KERNEL_ENTRY_POINTS(__threadsPerCore, __entry0, __entry1)                                              \
  namespace DeviceConfigNS {                                                                                           \
  extern constexpr DeviceConfig config{__threadsPerCore, KernelEntryPointFuncPtr(__entry0),                            \
                                       KernelEntryPointFuncPtr(__entry1)};                                             \
  static_assert(config.threadsPerCore == 1 || config.threadsPerCore == 2,                                              \
                "1 or 2 entry threads per core should be configured");                                                 \
  static_assert((config.threadsPerCore == 2 && (config.entryPoint_0 && config.entryPoint_1)) ||                        \
                  (config.threadsPerCore == 1 && (config.entryPoint_0 && !config.entryPoint_1)),                       \
                "Thread entry points should be consitent with threadsPerCore");                                        \
  }

int get_num_threads();
int get_relative_thread_id();

#endif
