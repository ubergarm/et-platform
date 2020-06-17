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
#define ONLY_FOR(cond) template <ElemKind U = elK, typename std::enable_if< cond, size_t>::type = 0>
  
template <ElemKind elK> class Writer {
  using T = typename elemKind2elemTy<elK>::type;
  const float scale_;
  const int32_t offset_;
  T* const ptr_;
public:
  Writer(T* ptr, float scale = 1.0, int32_t offset = 0 ) : scale_(scale), offset_(offset), ptr_(ptr)
  {}

  ONLY_FOR(U == Float16Ty)
  Writer &operator=(float value) {
    uint16_t v;
    dnn_lib::convertFp32ToFp16(value, v);
    *ptr_ = v;
    return *this;
  }

  ONLY_FOR( U == Int8QTy || U == UInt8QTy || U == Int16QTy || U == Int32QTy)
  Writer &operator=(float value) {
    *ptr_ = dnn_lib::quantize<T>(value, scale_, offset_);
    return *this;
  }

};


#undef ONLY_FOR

} // namespace dnn_lib

#endif /* WRITER_H */
