/*-------------------------------------------------------------------------
 * Copyright (C) 2019, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef ADDRESSER_H
#define ADDRESSER_H

#include "Float16.h"
#include "Writer.h"
#include "LibCommon.h"

namespace dnn_lib {

template <typename T> class Addresser {
  T *ptrT_;
  uint16_t *ptrfp16_;
  Writer<T> writer;
  float16 utilfp16;

  float scale_;
  int32_t offset_;

public:
  Addresser(void *ptr, float scale = 1.0, int32_t offset = 0) {
    if (std::is_same<T, float16>::value == true) {
      ptrfp16_ = (uint16_t *)ptr;
    } else if (std::is_same<T, int8_t>::value == true) {
      scale_ = scale;
      offset_ = offset;
      writer.scale_ = scale;
      writer.offset_ = offset;
      ptrT_ = (T *)ptr;
    } else if (std::is_same<T, uint8_t>::value == true) {
      scale_ = scale;
      offset_ = 0;
      writer.scale_ = scale;
      writer.offset_ = 0;
      ptrT_ = (T *)ptr;
    } else if (std::is_same<T, int16_t>::value == true) {
      scale_ = scale;
      offset_ = offset;
      writer.scale_ = scale;
      writer.offset_ = offset;
      ptrT_ = (T *)ptr;
    } else {
      ptrT_ = (T *)ptr;
    }
  }

  // READ
  template <typename U = T,
            typename std::enable_if<std::is_same<U, float16>::value,
                                    std::size_t>::type = 0>
  float operator[](const size_t index) const {
    float f;
    dnn_lib::convertFp16ToFp32(ptrfp16_[index], f);
    return f;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, float>::value,
                                    std::size_t>::type = 0>
  const T &operator[](const size_t index) const {
    return ptrT_[index];
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, int8_t>::value,
                                    std::size_t>::type = 0>
  float operator[](const size_t index) const {
    float i32 = dnn_lib::dequantize<int8_t>(ptrT_[index], scale_, offset_);
    return i32;
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, uint8_t>::value,
                                    std::size_t>::type = 0>
  float operator[](const size_t index) const {
    float i32 = dnn_lib::dequantize<uint8_t>(ptrT_[index], scale_, offset_);
    return i32;
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, int16_t>::value,
                                    std::size_t>::type = 0>
  float operator[](const size_t index) const {
    float i32 = dnn_lib::dequantize<int16_t>(ptrT_[index], scale_, offset_);
    return i32;
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, int64_t>::value,
                                    std::size_t>::type = 0>
  const T &operator[](const size_t index) const {
    return ptrT_[index];
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, int32_t>::value,
                                    std::size_t>::type = 0>
  const T &operator[](const size_t index) const {
    return ptrT_[index];
  }

  // write
  template <typename U = T,
            typename std::enable_if<std::is_same<U, float16>::value,
                                    std::size_t>::type = 0>
  Writer<T> &operator[](const size_t index) {
    writer.ptrfp16_ = &ptrfp16_[index];
    return writer;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, float>::value,
                                    std::size_t>::type = 0>
  T &operator[](const size_t index) {
    return ptrT_[index];
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, int8_t>::value,
                                    std::size_t>::type = 0>
  Writer<T> &operator[](const size_t index) {
    writer.ptri8_ = &ptrT_[index];
    return writer;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, uint8_t>::value,
                                    std::size_t>::type = 0>
  Writer<T> &operator[](const size_t index) {
    writer.ptrui8_ = &ptrT_[index];
    return writer;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, int16_t>::value,
                                    std::size_t>::type = 0>
  Writer<T> &operator[](const size_t index) {
    writer.ptri16_ = &ptrT_[index];
    return writer;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, int64_t>::value,
                                    std::size_t>::type = 0>
  T &operator[](const size_t index) {
    return ptrT_[index];
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, int32_t>::value,
                                    std::size_t>::type = 0>
  T &operator[](const size_t index) {
    return ptrT_[index];
  }
};

} // dnn_lib

#endif /* ADDRESSER_H */
