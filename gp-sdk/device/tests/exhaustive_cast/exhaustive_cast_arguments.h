#ifndef numElements
#define numElements 1000
#endif

#ifndef EXHAUSTIVE_CAST_ARGUMENTS_H
#define EXHAUSTIVE_CAST_ARGUMENTS_H
#define EXHAUSTIVE_CAST_VERIFICATION


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
  float a[numElements];
  int64_t b[numElements];
  uint64_t c[numElements];
  int32_t d[numElements];
  uint32_t e[numElements];
};

union outputContainer {
  float f[numElements];
  int64_t g[numElements];
  uint64_t h[numElements];
  int32_t i[numElements];
  uint32_t j[numElements];
};

struct KernelArguments {
  uint64_t cast_type;
  union inputContainer* in;
  union outputContainer* out;
} __attribute__ ((packed));

#endif

