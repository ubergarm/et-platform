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

#ifndef WRITER_H
#define WRITER_H

#include "Float16.h"
#include "LibCommon.h"

namespace dnn_lib {

template <typename T> class Writer {
public:
  uint16_t *ptrfp16_;
  int8_t *ptri8_;
  uint8_t *ptrui8_;
  int16_t *ptri16_;
  float scale_;
  int32_t offset_;

  template <typename U = T,
            typename std::enable_if<std::is_same<U, float16>::value,
                                    std::size_t>::type = 0>
  Writer &operator=(float value) {
    uint16_t v;
    dnn_lib::convertFp32ToFp16(value, v);
    *ptrfp16_ = v;
    return *this;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, int8_t>::value,
                                    std::size_t>::type = 0>
  Writer &operator=(float value) {
    *ptri8_ = dnn_lib::quantize<int8_t>(value, scale_, offset_);
    return *this;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, uint8_t>::value,
                                    std::size_t>::type = 0>
  Writer &operator=(float value) {
    *ptrui8_ = dnn_lib::quantize<uint8_t>(value, scale_, offset_);
    return *this;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, int16_t>::value,
                                    std::size_t>::type = 0>
  Writer &operator=(float value) {
    *ptri16_ = dnn_lib::quantize<int16_t>(value, scale_, offset_);
    return *this;
  }
};

} // namespace dnn_lib

#endif /* WRITER_H */
