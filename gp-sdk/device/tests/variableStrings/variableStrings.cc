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

#include "entryPoint.h"

class KernelArguments;
int entryPoint_0(KernelArguments* args);
void createString(int length, char *str);
DECLARE_KERNEL_ENTRY_POINTS(entryPoint_0, nullptr);

#define MAX_LENGTH 2048

void createString(int length, char *str) {
  // Limit the length to a maximum of 2048 characters
  length = (length < MAX_LENGTH) ? length : MAX_LENGTH - 1;

  // Fill the string with characters
  for (int i = 0; i < length; ++i) {
    str[i] = (char)('0' + (i % 10));  // Add consecutive digits '0' to '9'
  }
  str[length] = '\0';  // Null-terminate the string
}

int entryPoint_0([[maybe_unused]] KernelArguments* args) {
  if (get_relative_thread_id()==0) {
    for (int length = 2; length <= MAX_LENGTH; length *= 2) {
      char result[MAX_LENGTH];
      createString(length, result);
      et_printf("%s,%d\n","variableStrings", length);
      et_printf("%s", result);
    }
  }

  return 0;
}
