/**
 * Copyright (c) 2018-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLOAT16_H
#define FLOAT16_H

#define MAX_FP16_DENORM                                                        \
  ((float(1 << 10) - 1) /                                                      \
   float(1 << 24)) // maximum fp16 denormal = 2^-14 - 2^-24

/// Use a proxy type in case we need to change it in the future.
using Float16Storage = float;
class float16 {

public:
  float data_{ 0 };
  float16(uint16_t data) { data_ = *((float *)&data); }
  float16(float data) { data_ = data; }
  float16() { data_ = 0.0; }
  /*float16(float data = 0) {
    //if(data!=0.0)
      //toFp16(data,data_);
    //else
      data_= data;
  }*/

  /// Arithmetic operators.
  /* float16 operator*(const float16 &b) const {
     return float16(operator float() * float(b));
   }
   float16 operator/(const float16 &b) const {
     return float16(operator float() / float(b));
   }*/
  /*float16 operator+(const float16 &b){
    //return float16(operator float() + float(b));
    float16 tmp = fpAddSingleElement(data_,b.data_);
    return tmp;
  }*/
  /*float16 operator-(const float16 &b) const {
    return float16(operator float() - float(b));
  }
  float16 operator+=(const float16 &b) {
    *this = *this + b;
    return *this;
  }
  float16 operator-=(const float16 &b) {
    *this = *this - b;
    return *this;
  }*/

  float16 &operator=(float16 val) {
    data_ = val.data_;
    return *this;
  }
  float16 &operator=(float val) {
    data_ = val;
    return *this;
  }

  /*void store (unsigned int addr){
    __asm__ __volatile__
          (
          "mov.m.x m0, zero, 0x1 \n"
          "fcvt.f16.ps %[res], %[res] \n"
          "fsw.ps %[res], 0(%[addr]) \n"
          :[res] "=f" (data_)
          :[addr] "r" (addr)
          );
  }*/

  /*const float16 &operator[](unsigned int addr) const{
    __asm__ __volatile__
          (
          "mov.m.x m0, zero, 0x1 \n"
          "flw.ps f10, 0(%[addr]) \n"
          ://[ret] "=f" (data_)
          :[addr] "r" (addr)
          );
    return *this;
  }*/

  /*float16 &operator[](unsigned int addr) {
    __asm__ __volatile__
          (
          "mov.m.x m0, zero, 0x1 \n"
          "fcvt.f16.ps %[res], %[res] \n"
          "fsw.ps %[res], 0(%[addr]) \n"
          :[res] "=f" (data_)
          :[addr] "r" (addr)
          );
    return *this;
  }*/

  /* float16 const operator[](const unsigned int addr) const {
     float16 res;
     __asm__ __volatile__
           (
           "mov.m.x m0, zero, 0x1 \n"
           "flw.ps %[res], 0(%[addr]) \n"
           "fcvt.f16.ps %[res], %[res] \n"
           :[res] "=f" (res.data_)
           :[addr] "r" (addr)
           );
     return res;
   }*/
  /// Comparisons.
  bool operator<(const float16 &b) const { return this->data_ < b.data_; }
  bool operator>(const float16 &b) const { return this->data_ > b.data_; }
  bool operator==(const float16 &b) const { return this->data_ == b.data_; }
  bool operator>=(const float16 &b) const { return !(operator<(b)); }
  bool operator<=(const float16 &b) const { return !(operator>(b)); }

  /// Cast operators.
  // operator double() const { return double(operator float()); }
  operator float() const { return fp32_to_fp16_value(); }
  // operator float16() const { return fp32_to_fp16_value(); }
  // operator long long() const { return static_cast<long long>(data_); }

  // File stream operators
  /*friend std::ostream& operator<<(std::ostream& out, const float16 &b) {
    float outFloat = float(b);
    out << outFloat;
    return out;
  }*/

  float16 fp32_to_fp16_value() const {
    float16 ret;
    convertFp32ToFp16(data_, ret.data_);
    return ret;
  }

  float fp16_to_fp32_value() const {
    float dst = 0.0;
    convertFp16ToFp32(data_, dst);
    return dst;
  }

  inline
      __attribute__((always_inline)) void convertFp16ToFp32(float src,
                                                            float dst) const {
    __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                         "fcvt.ps.f16 %[dst], %[src] \n"
                         : [dst] "=f"(dst)
                         : [src] "f"(src));
  }
  inline __attribute__((always_inline)) float convertFp16ToFp32() const {
    float dst;
    __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                         "fcvt.ps.f16 %[dst], %[src] \n"
                         : [dst] "=f"(dst)
                         : [src] "f"(data_));
    return dst;
  }

  inline
      __attribute__((always_inline)) void toFp16(float src, float &dst) const {
    __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                         "fcvt.f16.ps %[dst], %[src] \n"
                         : [dst] "=f"(dst)
                         : [src] "f"(src));
  }
  inline
      __attribute__((always_inline)) void convertFp32ToFp16(float src,
                                                            float dst) const {
    __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                         "fcvt.f16.ps %[dst], %[src] \n"
                         : [dst] "=f"(dst)
                         : [src] "f"(src));
  }
  inline __attribute__((always_inline)) float convertFp32ToFp16() const {
    float dst;
    __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                         "fcvt.f16.ps %[dst], %[src] \n"
                         : [dst] "=f"(dst)
                         : [src] "f"(data_));
    return dst;
  }

  inline __attribute__((always_inline)) float fpAddSingleElement(
      float val1, float val2) const {
    float tmp1;
    float tmp2;
    float res;
    __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                         "fcvt.ps.f16 %[tmp1], %[val1] \n"
                         "fcvt.ps.f16 %[tmp2], %[val2] \n"
                         "fadd.ps %[res], %[tmp1], %[tmp2] \n"
                         "fcvt.f16.ps %[res], %[res] \n"
                         : [res] "=&f"(res), [tmp1] "=f"(tmp1),
                           [tmp2] "=f"(tmp2)
                         : [val1] "f"(val1), [val2] "f"(val2));
    return res;
  }

}; // End class float16.

#endif // GLOW_SUPPORT_FLOAT16_H
