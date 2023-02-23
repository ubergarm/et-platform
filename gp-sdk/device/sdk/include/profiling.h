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

#include "trace/trace_umode.h"

// this Workaround needs to be in place until proper System-software components are released
#define SAFE_PROFILING_INTEGRATION_WIP
#ifdef SAFE_PROFILING_INTEGRATION_WIP
#define et_trace_user_profile_event_unsafe(a, b, c, d) et_trace_user_profile_event(a, b, c, d)
#endif

/*! \file profiling.h
    \brief Functions and macros for tracing of user-defined events
*/

/**
 * \brief adapters to user_profile_events provided by tracing_subsystem.
 * Note at the moment function-name is part of the event identifier, so events should start and complete on the same
 * function.
 **/
template <bool safe> class ScopedUserProfileEvent {
public:
  ScopedUserProfileEvent() = delete;
  /*!
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
    if constexpr (safe) {
      et_trace_user_profile_event(regionId_, /*start*/ true, func_, line_);
    } else {
      et_trace_user_profile_event_unsafe(regionId_, /*start*/ true, func_, line_);
    }
  }

  /*!
   * Destroys ScopedUserProfileEvent after closing the event.
   * \brief Stops registering the event and destroys the object
   */
  ~ScopedUserProfileEvent() {
    if constexpr (safe) {
      et_trace_user_profile_event(regionId_, /*start*/ false, func_, line_);
    } else {
      et_trace_user_profile_event_unsafe(regionId_, /*start*/ false, func_, line_);
    }
  }

private:
  const char* func_ = nullptr;
  uint32_t line_ = 0;
  uint16_t regionId_ = 0;
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
    PMC's are read in safe mode (supports simultaneous profiling from the 2 threads on the minon).
  - Event begins when where code is placed.
  - Event ends where the current C++ scope ends
  \param regionId region to instrument
*/
#define SCOPED_USER_PROFILE_EVENT(regionId)                                                                            \
  ScopedUserProfileEvent<true> UNIQUE(scopedUserProfileEvent)(__func__, __LINE__, regionId)

/*!
  \def SCOPED_USER_PROFILE_EVENT_FAST(regionId)
  \brief Generates a scoped user profiling event identified with region. function name and line will also be quoted.
   PMC's are *not* read in safe mode (sporadic counter corruptions may happen if simultaneously frofiling from the 2
  threads on the Minion) minoin collide.
  - Event begins when where code is placed.
  - Event ends where the current C++ scope ends
  \param regionId region to instrument
*/
#define SCOPED_USER_PROFILE_EVENT_FAST(regionId)                                                                       \
  ScopedUserProfileEvent<false> UNIQUE(scopedUserProfileEvent)(__func__, __LINE__, regionId)
/*!
  \def USER_PROFILE_EVENT_START_FAST(regionId)
  \brief
  Generates the start of a user profiling event. function name and line will also be quoted.
  This event needs to be expcicitlly completed (USER_PROFILE_EVENT_END of same regionId)
  PMC's are *not* read in safe mode (sporadic counter corruptions may happen if simultaneously frofiling from the 2
  threads on the Minion) threads in the minion collide \param regionId region to instrument
*/
#define USER_PROFILE_EVENT_START_FAST(regionId)                                                                        \
  do {                                                                                                                 \
    et_trace_user_profile_event_unsafe(regionId, true, __func__, __LINE__);                                            \
  } while (0)

/*!
  \def  USER_PROFILE_EVENT_START(regionId)
  \brief
  Generates the start of a user profiling event. function name and line will also be quoted.
  This event needs to be expcicitlly completed (USER_PROFILE_EVENT_END of same regionId)
  PMC's are read in safe mode (supports simultaneous profiling from the 2 threads on the minon).
  \param regionId region to instrument
 */
#define USER_PROFILE_EVENT_START(regionId)                                                                             \
  do {                                                                                                                 \
    et_trace_user_profile_event(regionId, true, __func__, __LINE__);                                                   \
  } while (0)

/*!
 \def  USER_PROFILE_EVENT_END_FAST(regionId)
 \brief Completes a profiling event. Function name and line will also be quoted.
  PMC's are *not* read in safe mode (sporadic counter corruptions may happen if simultaneously frofiling from the 2
 threads on the Minion) threads in the midion collide \param regionId region to instrument \param regionId region to
 instrument
*/
#define USER_PROFILE_EVENT_END_FAST(regionId)                                                                          \
  do {                                                                                                                 \
    et_trace_user_profile_event_unsafe(regionId, false, __func__, __LINE__);                                           \
  } while (0)

/*!
  \def  USER_PROFILE_EVENT_END(regionId)
  \brief Completes a profiling event. Function name and line will also be quoted.
   PMC's are read in safe mode (supports simultaneous profiling from the 2 threads on the minon).
  \param regionId region to instrument
*/
#define USER_PROFILE_EVENT_END(regionId)                                                                               \
  do {                                                                                                                 \
    et_trace_user_profile_event(regionId, false, __func__, __LINE__);                                                  \
  } while (0)

#endif // GPSDK_PROFILING_H
