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

#include "Float16.h"
#include "LibCommon.h"

float16::float16(uint16_t data) {
  uint32_t data32 = static_cast<uint32_t>(data);
  data_ = dnn_lib::bitwise_copy<float>(data32);
}

float16::float16(float data) { data_ = data; }

float16::float16() { data_ = 0.0; }

float16 & float16::operator=(float16 val) {
  data_ = val.data_;
  return *this;
}

float16 & float16::operator=(float val) {
  data_ = val;
  return *this;
}

bool float16::operator<(const float16 &b) const { return this->data_ < b.data_; }
bool float16::operator>(const float16 &b) const { return this->data_ > b.data_; }
bool float16::operator==(const float16 &b) const { return this->data_ == b.data_; }
bool float16::operator>=(const float16 &b) const { return !(operator<(b)); }
bool float16::operator<=(const float16 &b) const { return !(operator>(b)); }

float16::operator float() const { return fp32_to_fp16_value(); }

float16 float16::fp32_to_fp16_value() const {
  float16 ret;
  dnn_lib::convertFp32ToFp16(data_, ret.data_);
  return ret;
}

float float16::fp16_to_fp32_value() const {
  float dst = 0.0;
  dnn_lib::convertFp16ToFp32(data_, dst);
  return dst;
}

inline __attribute__((always_inline)) float float16::convertFp32ToFp16() const {
  float dst;
  dnn_lib::convertFp32ToFp16(data_, dst);
  return dst;
}

inline __attribute__((always_inline)) float float16::fpAddSingleElement(
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
