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

#include <etsoc/common/utils.h>
#include <etsoc/isa/hart.h>
#include <algorithm>

#include "entryPoint.h"
#include "profiling.h"

class KernelArguments;
int entryPoint_0(KernelArguments* vectors);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

const char MYREGIONNAME[] = "myRegion1";

int entryPoint_0([[ maybe_unused ]] KernelArguments* vectors) {

  if(get_relative_thread_id() != 0) {
   return 0;
  }
  //this test assumes ~4096 bytes per hart trace buffer.
  //1000 is enough to stress the circular buffer wrap-around (4096 bytes) few times.
  //goal is getting all the writes
  constexpr size_t kMaxIters = 1000;
  for(size_t i = 0; i < kMaxIters; i ++) {

    SCOPED_USER_PROFILE_EVENT("first_region");
    SCOPED_USER_PROFILE_EVENT("second_region");
    SCOPED_USER_PROFILE_EVENT("third_region");
    SCOPED_USER_PROFILE_EVENT(1);
    SCOPED_USER_PROFILE_EVENT(2);

    USER_PROFILE_EVENT_START(3);
    USER_PROFILE_EVENT_START(4);
    USER_PROFILE_EVENT_START("user_event_start");
    USER_PROFILE_EVENT_START("user_event_start2");

    SCOPED_USER_PROFILE_EVENT(MYREGIONNAME);

    USER_PROFILE_EVENT_END("user_event_start2");
    USER_PROFILE_EVENT_END("user_event_start");
    USER_PROFILE_EVENT_END(4);
    USER_PROFILE_EVENT_END(3);
  }

  return 0;
}
