/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef NEURALIZER_DEVICE_TYPES_H
#define NEURALIZER_DEVICE_TYPES_H

#include <assert.h>
#include <stdint.h>

namespace neura {

// Role of a Hart
enum class HartRole {
  // Perform computations
  COMPUTE,
  // Prefetch activations
  ACT_PREF,
  // Prefetch weights
  W_PREF,
  // Prefetch instructions
  INST_PREF,
  // Synchronization
  SYNC,
  // Total numer of roles
  TOTAL
};

// Data that is recorded for each node when tracing nodes
struct TracingData {
  // Node starting cycle after synchronization
  uint64_t syncStartTimestamp;
  // Node ending cycle without synchronization
  uint64_t asyncEndTimestamp;
};

// Returns a string that represents the value of a hart role
static inline char const* toString(HartRole hartRole) {
  switch (hartRole) {
  case HartRole::COMPUTE:
    return "compute";
  case HartRole::ACT_PREF:
    return "act_pref";
  case HartRole::W_PREF:
    return "w_pref";
  case HartRole::INST_PREF:
    return "inst_pref";
  case HartRole::SYNC:
    return "sync";
  case HartRole::TOTAL:
    assert(false and "Invalid hart role");
    return "";
  }

  assert(false and "unknown hart role");
  return nullptr;
}

} // namespace neura

#endif // NEURALIZER_DEVICE_TYPES_H
