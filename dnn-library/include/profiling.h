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
#ifndef GPSDK_PROFILING_H
#define GPSDK_PROFILING_H

#include "GPSDKCommonCode.h"
#include "trace/trace_umode.h"
#include <etsoc/common/utils.h>

/*! \file profiling.h
    \brief Functions and macros for tracing of user-defined events
*/

static inline void etTraceUserProfileEvent(uint16_t regionId, bool start, const char* func, uint32_t line) {
  et_trace_user_profile_event(regionId, start, func, line, nullptr)
}

static inline void etTraceUserProfileEvent(const char* regionId, bool start, const char* func, uint32_t line) {
  et_trace_user_profile_event(0, start, func, line, regionId)
}

/**
 * \brief adapters to user_profile_events provided by tracing_subsystem.
 * Note at the moment function-name is part of the event identifier, so events should start and complete on the same
 * function.
 **/
class ScopedUserProfileEvent {
public:
  ScopedUserProfileEvent() = delete;
  /**
   * Creates a new ScopedUserProfileEvent object that inmediately is registered as an event
   * that has a function name, line of code, and regionId defined by the user.
   * \brief Main constructor, start of event.
   * \param func string specifying function name
   * \param line line of code
   * \param regionId user-defined numeric Id
   */
  ScopedUserProfileEvent(const char* func, uint32_t line, uint16_t regionId)
    : func_(func)
    , line_(line)
    , regionId_(regionId) {
    et_trace_user_profile_event(regionId_, /*start*/ true, func_, line_, regname_);
  }

  ScopedUserProfileEvent(const char* func, uint32_t line, const char* regionName)
    : func_(func)
    , line_(line)
    , regname_(regionName) {
    et_trace_user_profile_event(regionId_, /*start*/ true, func_, line_, regname_);
  }

  /**
   * Destroys ScopedUserProfileEvent after closing the event.
   * \brief Stops registering the event and destroys the object
   */
  ~ScopedUserProfileEvent() {
    et_trace_user_profile_event(regionId_, /*start*/ false, func_, line_, regname_);
  }

private:
  const char* func_ = nullptr;
  uint32_t line_ = 0;
  uint16_t regionId_ = 0;
  const char* regname_ = nullptr;
};

/*! \cond PRIVATE */
/** helpers to create a unique name quoting the line */
#define CONCAT(a, b) a##b
#define UNIQUE_1(i, n) CONCAT(i, n)
#define UNIQUE(name) UNIQUE_1(name, __LINE__)
/*! \endcond */

/*!
  \def SCOPED_USER_PROFILE_EVENT(regionId)
  \brief Generates a scoped user profiling event identified with region. function name and line will also be quoted.
  - Event begins when where code is placed.
  - Event ends where the current C++ scope ends
  \param regionId region to instrument
*/
#define SCOPED_USER_PROFILE_EVENT(regionId)                                                                            \
  ScopedUserProfileEvent UNIQUE(scopedUserProfileEvent)(__func__, __LINE__, regionId)

/**
 * \brief
 * Generates the start of a user profiling event. function name and line will also be quoted.
 * This event needs to be expcicitlly completed (USER_PROFILE_EVENT_END of same regionId)
 * \param regionId region to instrument
 */
#define USER_PROFILE_EVENT_START(regionId)                                                                             \
  do {                                                                                                                 \
    etTraceUserProfileEvent(regionId, true, __func__, __LINE__);                                                       \
  } while (0)

/**
 * \brief Completes a profiling event. Function name and line will also be quoted.
 * \param regionId region to instrument
 */
#define USER_PROFILE_EVENT_END(regionId)                                                                               \
  do {                                                                                                                 \
    etTraceUserProfileEvent(regionId, false, __func__, __LINE__);                                                      \
  } while (0)

/**
 * \internal
 * \brief Mark a span of code to be timed. The timing result will be printed to the trace.
 */
struct ScopedTimedRegion {
  ScopedTimedRegion() = delete;
  ScopedTimedRegion(ScopedTimedRegion&) = delete;
  ScopedTimedRegion(ScopedTimedRegion&&) = delete;
  ScopedTimedRegion& operator=(const ScopedTimedRegion&) = delete;
  ScopedTimedRegion& operator=(ScopedTimedRegion&&) = delete;

  /**
   * \internal
   * \brief Begin the timed span providing a location and name.
   *
   * Use SCOPED_TIMED_REGION()
   */
  ScopedTimedRegion(const char* name, const char* func, unsigned int line, bool allThreads = false)
    : name_(name)
    , func_(func)
    , line_(line)
    , allThreads_(allThreads){};

  ~ScopedTimedRegion() {
    auto totalTime = et_get_delta_timestamp(startTimestamp_);
    if (allThreads_ || get_hart_id() == 0)
      et_printf("[%u - %s:%u] region %s took %lu cycles (%.3f us)\n", get_hart_id(), func_, line_, name_, totalTime,
                double(cyclesToNs(totalTime)) / 1000.0);
  };

private:
  uint64_t startTimestamp_ = et_get_timestamp();
  const char* name_;
  const char* func_;
  unsigned int line_;
  bool allThreads_;
};

/**
 * \brief Mark a region of code to be timed. The time will appear in the trace.
 *
 * The time will be printed to the trace, with the region name, function name, and line number.
 *  - The region begins when where code is placed.
 *  - The region ends where the current C++ scope ends
 *
 * This is useful for simple benchmarking tasks. It doesn't replace real benchmarking, but it eliminates the
 * need for explicit "get start time, get stop time, then print the difference" code.
 *
 * \param name The name of the region
 */
#define SCOPED_TIMED_REGION(name) ScopedTimedRegion UNIQUE(scopedTimedRegion)(name, __func__, __LINE__)

#define SCOPED_TIMED_REGION_ALL(name) ScopedTimedRegion UNIQUE(scopedTimedRegion)(name, __func__, __LINE__, true)

#endif // GPSDK_PROFILING_H
