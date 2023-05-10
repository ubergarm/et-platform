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

#ifndef ADDRESSER_V2_H
#define ADDRESSER_V2_H

#include "Float16.h"
#include "LibCommon.h"
#include "LibTypes.h"
#include "internal_v2/Writer.h"

namespace dnn_lib_v2 {

#define ONLY_FOR(cond) template <ElemKind U = elK, typename std::enable_if<(cond), size_t>::type = 0>

template <ElemKind elK, bool globalStore = false> class Addresser {
  using T = typename elemKind2elemTy<elK>::type;

public:
  Addresser(void* ptr, float scale = 1.0, int32_t offset = 0)
    : scale_(scale)
    , offset_(offset)
    , ptr_(reinterpret_cast<T*>(ptr)) {
  }

  ////////////////////////////////////////////////////////////////////////////////
  // READ
  ////////////////////////////////////////////////////////////////////////////////

  // Float16 => converts to float
  ONLY_FOR(U == Float16Ty)
  float operator[](const size_t index) const {
    float f;
    dnn_lib::convertFp16ToFp32(ptr_[index], f);
    return f;
  }

  // Integer quantized types: converts to float
  ONLY_FOR(U == Int8QTy || U == UInt8QTy || U == Int16QTy || U == Int32QTy)
  float operator[](const size_t index) const {
    return dnn_lib::dequantize<T>(ptr_[index], scale_, offset_);
  }

  // none of the above cases (Float, index types...) read directly from ptr
  ONLY_FOR(U != Float16Ty && U != Int8QTy && U != UInt8QTy && U != Int16QTy && U != Int32QTy)
  T operator[](const size_t index) const {
    return ptr_[index];
  }

  ////////////////////////////////////////////////////////////////////////////////
  // WRITE
  ////////////////////////////////////////////////////////////////////////////////

#define SUPPORT_WRITER(U)                                                                                              \
  ((U == FloatTy) || (U == Float16Ty) || (U == BFloat16Ty) || (U == Int8QTy) || (U == UInt8QTy) || (U == Int16QTy) ||  \
   (U == Int32QTy) || U == Int32ITy || U == Int64ITy)
  // quantized and float16: return a writer to write via float
  ONLY_FOR(SUPPORT_WRITER(U))
  Writer<elK, globalStore> operator[](const size_t index) {
    return Writer<elK, globalStore>(ptr_ + index, scale_, offset_);
  }

  // other types, direct write: return reference
  ONLY_FOR(!SUPPORT_WRITER(U))
  T& operator[](const size_t index) {
    // Sanity: only Writer knows about globalStore.
    static_assert(!globalStore, "globalStore not supported for this type in Addresser");
    return ptr_[index];
  }

#undef SUPPORT_WRITER

  ////////////////////////////////////////////////////////////////////////////////
  // Access
  ////////////////////////////////////////////////////////////////////////////////
  float getScale() const {
    return scale_;
  }

  int32_t getOffset() const {
    return offset_;
  }

private:
  const float scale_;
  const int32_t offset_;
  T* ptr_;
};

#undef ONLY_FOR

} // namespace dnn_lib_v2

#endif /* ADDRESSER_V2_H */