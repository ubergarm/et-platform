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

/**
 * Entry point for user code, from all threads allocated, only those with get_thread_id()==0
 * will enter this function.
 *
 * \brief main() for all threads with thread_id==0
 *
 * \param args struct of type KernelArguments
 * \param value Number of threads to synchronize. Must be a multiple of 32 or a power of two <= 32.
 * \param num_bytes number of bytes to write
 */
extern "C" int entryPoint_0(KernelArguments* args);

/**
 * Copies the 32-bit \p value into each of the first \p num_bytes characters of the object pointed to by \p ptr.
 * \brief main() for all threads with thread_id==1
 *
 * \param ptr First minion in the synchronization group. Must be a multiple of \p count or 0.
 * \param value Number of threads to synchronize. Must be a multiple of 32 or a power of two <= 32.
 * \param num_bytes number of bytes to write
 */
extern "C" int entryPoint_1(KernelArguments* args);

/* EntryPoint kernel functions must have this form */
using KernelEntryPointFuncPtr = int (*)(KernelArguments*);

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
