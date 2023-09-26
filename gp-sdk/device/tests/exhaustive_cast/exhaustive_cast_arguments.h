#ifndef numElements
#define numElements 1000
#endif

#ifndef EXHAUSTIVE_CAST_ARGUMENTS_H
#define EXHAUSTIVE_CAST_ARGUMENTS_H
// #define EXHAUSTIVE_CAST_VERIFICATION


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

union inputContainer  {
  float a[1000];
  int64_t b[1000];
  uint64_t c[1000];
  int32_t d[1000];
  uint32_t e[1000];
};

union outputContainer {
  float f[1000];
  int64_t g[1000];
  uint64_t h[1000];
  int32_t i[1000];
  uint32_t j[1000];
};

struct KernelArguments {
  uint64_t cast_type;
  union inputContainer* in;
  union outputContainer* out;
} __attribute__ ((packed));

#endif

