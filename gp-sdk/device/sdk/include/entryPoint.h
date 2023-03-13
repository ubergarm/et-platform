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

/**
 * Forward declaration of KernelArguments, user implementations will define
 */
class KernelArguments;

/* EntryPoint kernel functions must have this form */
using KernelEntryPointFuncPtr = int (*)(KernelArguments*);

KernelEntryPointFuncPtr rebaseEntryPointPtr(KernelEntryPointFuncPtr fnc, uint64_t base);

/// @brief __DeviceConfig
/// Examples of DeviceConfig
/// extern DeviceConfig config1 {1, entryPoint_0, nullptr};
/// extern DeviceConfig config2 {2, entryPoint_0, entryPoint_0};
/// extern eviceConfig config3 {2, entryPoint_0, entryPoint_1};
struct alignas(64) __DeviceConfig {
  int32_t threadsPerCore = 0;
  KernelEntryPointFuncPtr entryPoint_0 = nullptr;
  KernelEntryPointFuncPtr entryPoint_1 = nullptr;
};

// @brief DeviceConfig: type alias to enforce const definition. 
using DeviceConfig = const __DeviceConfig;

int get_num_threads();
int get_relative_thread_id();

#endif 
