
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

#include "LibCommon.h"

#define MAX_FP16_DENORM                                                        \
  ((float(1 << 10) - 1) /                                                      \
   float(1 << 24)) // maximum fp16 denormal = 2^-14 - 2^-24

/// Use a proxy type in case we need to change it in the future.
using Float16Storage = float;
class float16 {

public:
  float data_{ 0 };
  
  float16(uint16_t data) {
    uint32_t data32 = static_cast<uint32_t>(data);
    data_ = dnn_lib::bitwise_copy<float>(data32);
  }

  float16(float data) { data_ = data; }

  float16() { data_ = 0.0; }

  float16 &operator=(float16 val) {
    data_ = val.data_;
    return *this;
  }

  float16 &operator=(float val) {
    data_ = val;
    return *this;
  }

  /// Comparisons.
  bool operator<(const float16 &b) const { return this->data_ < b.data_; }
  bool operator>(const float16 &b) const { return this->data_ > b.data_; }
  bool operator==(const float16 &b) const { return this->data_ == b.data_; }
  bool operator>=(const float16 &b) const { return !(operator<(b)); }
  bool operator<=(const float16 &b) const { return !(operator>(b)); }

  /// Cast operators.

  operator float() const { return fp32_to_fp16_value(); }

  float16 fp32_to_fp16_value() const {
    float16 ret;
    dnn_lib::convertFp32ToFp16(data_, ret.data_);
    return ret;
  }

  float fp16_to_fp32_value() const {
    float dst = 0.0;
    dnn_lib::convertFp16ToFp32(data_, dst);
    return dst;
  }

  inline __attribute__((always_inline)) float convertFp32ToFp16() const {
    float dst;
    dnn_lib::convertFp32ToFp16(data_, dst);
    return dst;
  }

  inline __attribute__((always_inline)) float fpAddSingleElement(
      float a, float b) const {

    __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                         "fcvt.ps.f16 %[a], %[a] \n"
                         "fcvt.ps.f16 %[b], %[b] \n"
                         "fadd.ps %[b], %[b], %[a] \n"
                         "fcvt.f16.ps %[b], %[b] \n"
                         : [a] "+&f"(a),
                           [b] "+&f"(b)
                         );

    return b;
  }

}; // End class float16.

#endif // GLOW_SUPPORT_FLOAT16_H
