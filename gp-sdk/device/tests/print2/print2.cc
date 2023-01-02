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

#include <array>
#include <stdio.h>

#include <etsoc/common/utils.h>
#include <etsoc/isa/fcc.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <etsoc/isa/utils.h>

#include "entryPoint.h"

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  if (get_minion_id()==0) {
    et_printf("%s,%d HELLO WORLD!!!!\n",__func__,__LINE__);
    et_printf("%s,%d PRINT2 test executed!!!!\n",__func__,__LINE__);      
  }
  return 0;
}
