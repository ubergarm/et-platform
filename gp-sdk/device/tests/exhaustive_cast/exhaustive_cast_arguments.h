#ifndef EXHAUSTIVE_CAST_ARGUMENTS_H
#define EXHAUSTIVE_CAST_ARGUMENTS_H
#ifndef EXHAUSTIVE_CAST_VERIFICATION
#define EXHAUSTIVE_CAST_VERIFICATION
#endif

/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
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
