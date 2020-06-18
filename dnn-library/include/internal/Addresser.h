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

#define ONLY_FOR(cond) template <ElemKind U = elK, typename std::enable_if< (cond), size_t>::type = 0>

  
template <ElemKind elK> class Addresser {
  using T = typename elemKind2elemTy<elK>::type;
public:
  Addresser(void *ptr, float scale = 1.0, int32_t offset = 0):
    scale_(scale),
    offset_(offset),
    ptr_( reinterpret_cast<T*>(ptr) )
  { }

  ////////////////////////////////////////////////////////////////////////////////
  // READ
  ////////////////////////////////////////////////////////////////////////////////
  // direct read from pointer
  ONLY_FOR( U == FloatTy || U == Int64ITy || U == Int32ITy || U == BoolTy)
  const T operator[](const size_t index) const {
    return ptr_[index];
  }

  // Float16 => converts to float 
  ONLY_FOR( U == Float16Ty)
  const float operator[](const size_t index) const {
    float f;
    dnn_lib::convertFp16ToFp32(ptr_[index], f);
    return f;
  }

  // Integer quantized types: converts to float
  ONLY_FOR( U == Int8QTy || U == UInt8QTy || U == Int16QTy || U == Int32QTy)
  const float operator[](const size_t index) const {
    return dnn_lib::dequantize<T>(ptr_[index], scale_, offset_);
  }

  
  ////////////////////////////////////////////////////////////////////////////////
  // WRITE
  ////////////////////////////////////////////////////////////////////////////////

  // direct write: return reference
  ONLY_FOR( U == FloatTy  || U == Int64ITy || U == Int32ITy || U == BoolTy)
  T &operator[](const size_t index) {
    return ptr_[index];
  }

  // other types: return a writer to write via float
  ONLY_FOR( U == Float16Ty || U == Int8QTy || U == UInt8QTy || U == Int16QTy || U == Int32QTy)
  Writer<elK> operator[](const size_t index) {
    return Writer<elK>(ptr_ + index);
  }


private:
  const float scale_;
  const int32_t offset_;
  T* ptr_;
  
};

#undef ONLY_FOR
  
} // dnn_lib

#endif /* ADDRESSER_H */
