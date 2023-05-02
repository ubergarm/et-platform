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

#ifndef DENOISE_MASKS_H_
#define DENOISE_MASKS_H_

#include "0_filterMask.h"
#include "1_filterMask.h"
#include "2_filterMask.h"
#include "3_filterMask.h"
#include "4_filterMask.h"
#include "5_filterMask.h"
#include "6_filterMask.h"
#include "7_filterMask.h"
#include "8_filterMask.h"
#include "9_filterMask.h"
#include "LibTypes.h"
#include <array>

namespace dnn_lib {
namespace inlining {

// Automatically generated with ./noise_denoise:

static constexpr auto kFilterSize = 256 * 256;

// initialize filter bank with auto-generated filters.

alignas(CACHE_LINE_BYTES) static constexpr std::array<std::array<uint8_t, kFilterSize>, 10> denoiseMask = {
  f_0_filterMask__amp1_21__amp2_26__bw1_10__bw2_5__lim1_37__lim3_44,
  f_1_filterMask__amp1_24__amp2_67__bw1_9__bw2_3__lim1_65__lim3_57,
  f_2_filterMask__amp1_30__amp2_54__bw1_8__bw2_4__lim1_86__lim3_43,
  f_3_filterMask__amp1_41__amp2_66__bw1_8__bw2_3__lim1_87__lim3_65,
  f_4_filterMask__amp1_45__amp2_75__bw1_8__bw2_4__lim1_96__lim3_31,
  f_5_filterMask__amp1_54__amp2_70__bw1_10__bw2_6__lim1_37__lim3_47,
  f_6_filterMask__amp1_55__amp2_55__bw1_23__bw2_14__lim1_80__lim3_35,
  f_7_filterMask__amp1_66__amp2_43__bw1_9__bw2_4__lim1_87__lim3_38,
  f_8_filterMask__amp1_66__amp2_71__bw1_8__bw2_10__lim1_36__lim3_89,
  f_9_filterMask__amp1_73__amp2_22__bw1_8__bw2_5__lim1_26__lim3_41};

} // namespace inlining

} // namespace dnn_lib

#endif // DENOISE_MASKS_H_