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

/// @brief hello
/// Examples of DeviceConfig
/// constexpr DeviceConfig config1 {1, entryPoint_0, nullptr};
/// constexpr DeviceConfig config2 {2, entryPoint_0, entryPoint_0};
/// constexpr DeviceConfig config3 {2, entryPoint_0, entryPoint_1};
struct DeviceConfig {
  int32_t threadsPerCore = 0;
  KernelEntryPointFuncPtr entryPoint_0 = nullptr;
  KernelEntryPointFuncPtr entryPoint_1 = nullptr;
};

int get_num_threads();
int get_relative_thread_id();

#endif 
