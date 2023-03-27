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

#include <cstdint>

/* EntryPoint kernel function prototype
 * note that user entry points can receive typed ptr (namely KernelArguments *)*/
using KernelEntryPointFuncPtr = int (*)(void*);

/// @brief __DeviceConfig

struct alignas(64) __DeviceConfig {
  int32_t threadsPerCore = 0;
  bool sameEntryPoint;
  KernelEntryPointFuncPtr entryPoint_0 = nullptr;
  KernelEntryPointFuncPtr entryPoint_1 = nullptr;
};

/// @brief DeviceConfig: type alias to enforce const definition.
using DeviceConfig = const __DeviceConfig;

namespace device_config {
template <typename T0, typename T1> static constexpr int32_t getThreadsPerCore(T0 e0, T1 e1) {
  return e0 ? (e1 ? 2 : 1) : 0;
}

// Note: gcc 8.2 refuses to constexpr-compare function pointers, so we use partial template specialization as a
// workaround. See  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77911

// Partial specializatoin when f1!=f2
template <typename T, T f1, T f2> struct IsSameFuncPtr { static constexpr bool value = false; };

// Partial Specialization when f1==f2
template <typename T, T f> struct IsSameFuncPtr<T, f, f> { static constexpr bool value = true; };

} // namespace device_config

/// @brief macro for registering a single DeviceConfig singleton
/// \param entry0 entryPoint for ET-Minion thread 0 (Note, If only 1 ThreadPerCore is in use, it should be 0)
/// \param entry1 entryPoint for ET-Minion thread 1
//
// Examples:
// /// Examples of DeviceConfig
/// DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);
/// DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, entryPoint_1);
/// DECLARE_KERNEL_ENTRY_POINTS(entryPoint, entryPoint);
#define DECLARE_KERNEL_ENTRY_POINTS(__entry0, __entry1)                                                                \
  namespace device_config {                                                                                            \
  extern constexpr DeviceConfig config{getThreadsPerCore(__entry0, __entry1),                                          \
                                       IsSameFuncPtr<decltype(&__entry0), __entry0, __entry1>::value,                  \
                                       KernelEntryPointFuncPtr(__entry0), KernelEntryPointFuncPtr(__entry1)};          \
  static_assert(config.threadsPerCore != 0,                                                                            \
                "1 or 2 Threads per core should be configured (in case of 1 it should be thread 0)");                  \
  }

int get_num_threads();
int get_relative_thread_id();
int get_relative_thread_id(uint64_t shireMask);

#endif
