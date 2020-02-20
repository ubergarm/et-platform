/*-------------------------------------------------------------------------
* Copyright (C) 2018, Esperanto Technologies Inc.
* The copyright to the computer program(s) herein is the
* property of Esperanto Technologies, Inc. All Rights Reserved.
* The program(s) may be used and/or copied only with
* the written permission of Esperanto Technologies and
* in accordance with the terms and conditions stipulated in the
* agreement/contract under which the program(s) have been supplied.
*-------------------------------------------------------------------------
*/

#ifndef FLOAT16_H
#define FLOAT16_H

#include <cstdint>

static const float MAX_FP16_DENORM = ((float(1 << 10) - 1) / float(1 << 24)); // maximum fp16 denormal = 2^-14 - 2^-24

/// Use a proxy type in case we need to change it in the future.

using Float16Storage = float;

class float16 {
public:

  float data_{ 0 };
  
  float16(uint16_t data);

  float16(float data);

  float16();

  float16 &operator=(float16 val);

  float16 &operator=(float val);

  /// Comparisons.
  bool operator<(const float16 &b) const;
  bool operator>(const float16 &b) const;
  bool operator==(const float16 &b) const;
  bool operator>=(const float16 &b) const;
  bool operator<=(const float16 &b) const;

  /// Cast operators.

  operator float() const;

  float16 fp32_to_fp16_value() const;

  float fp16_to_fp32_value() const;

  inline __attribute__((always_inline)) float convertFp32ToFp16() const;

  inline __attribute__((always_inline)) float fpAddSingleElement(float a, float b) const;

};

#endif // GLOW_SUPPORT_FLOAT16_H
