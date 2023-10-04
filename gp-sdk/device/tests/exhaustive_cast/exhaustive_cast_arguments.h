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

enum CastType {
  floatToInt64_t = 1,
  floatToUint64_t = 2,
  floatToInt32_t = 3,
  floatToUint32_t = 4,
  int64_tToFloat = 5,
  uint64_tToFloat = 6,
  int32_tToFloat = 7,
  uint32_tToFloat = 8
};

constexpr size_t numElements = 1000;
union dataContainer {
  float a[numElements];
  int64_t b[numElements];
  uint64_t c[numElements];
  int32_t d[numElements];
  uint32_t e[numElements];
};

struct KernelArguments {
  CastType cast_type;
  union dataContainer* in;
  union dataContainer* out;
} __attribute__((packed));

#endif
