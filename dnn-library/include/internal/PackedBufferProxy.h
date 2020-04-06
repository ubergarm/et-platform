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

#ifndef PACKED_BUFFER_PROXY_H
#define PACKED_BUFFER_PROXY_H

#include "Float16.h"
#include "Writer.h"
#include "LibCommon.h"

template <typename T>
class PackedBufferProxy
{ };

template <>
class PackedBufferProxy<float> {
private:
  float *ptr_;

public:
  PackedBufferProxy<float>(void* ptr) : ptr_{static_cast<float*>(ptr)} {}

  float get(size_t index) const { return ptr_[index]; }

  void set(size_t index, float value) const { ptr_[index] = value; }

  float unpack(size_t index) const { return ptr_[index]; }

  void pack(size_t index, float value) const { ptr_[index] = value; }
};

template <>
class PackedBufferProxy<float16> {
private:
  uint16_t *ptr_;

public:
  PackedBufferProxy<float16>(void* ptr) : ptr_{static_cast<uint16_t*>(ptr)} {}

  uint16_t get(size_t index) const { return ptr_[index]; }

  void set(size_t index, uint16_t value) const { ptr_[index] = value; }

  float unpack(size_t index) const {
    float floatValue;
    dnn_lib::convertFp16ToFp32(ptr_[index], floatValue);
    return floatValue;
  }

  void pack(size_t index, float value) const {
    uint16_t uint16Value;
    dnn_lib::convertFp32ToFp16(value, uint16Value);
    ptr_[index] = uint16Value;
  }
};

template <typename T>
class IntegerPackedBufferProxy {
private:
  T *ptr_;
  float scale_;
  int32_t offset_;

public:
  IntegerPackedBufferProxy(void* ptr, float scale, int32_t offset) :
    ptr_(static_cast<T*>(ptr)),
    scale_{scale},
    offset_{offset}
  { }

  T get(size_t index) const { return ptr_[index]; }

  void set(size_t index, T value) const { ptr_[index] = value; }

  float unpack(size_t index) const {
    return dnn_lib::dequantize<T>(ptr_[index], scale_, offset_);
  }

  void pack(size_t index, float value) const {
    ptr_[index] = dnn_lib::quantize<T>(value, scale_, offset_);
  }
};

template <>
class PackedBufferProxy<int8_t> : IntegerPackedBufferProxy<int8_t> {
public:
  PackedBufferProxy<int8_t>(void* ptr, float scale, int32_t offset) :
    IntegerPackedBufferProxy<int8_t>(ptr, scale, offset)
  { }
};

template <>
class PackedBufferProxy<uint8_t> : IntegerPackedBufferProxy<uint8_t> {
public:
  PackedBufferProxy<uint8_t>(void* ptr, float scale, int32_t offset) :
    IntegerPackedBufferProxy<uint8_t>(ptr, scale, offset)
  { }
};

template <>
class PackedBufferProxy<int16_t> : IntegerPackedBufferProxy<int16_t> {
public:
  PackedBufferProxy<int16_t>(void* ptr, float scale, int32_t offset) :
    IntegerPackedBufferProxy<int16_t>(ptr, scale, offset)
  { }
};

template <>
class PackedBufferProxy<uint16_t> : IntegerPackedBufferProxy<uint16_t> {
public:
  PackedBufferProxy<uint16_t>(void* ptr, float scale, int32_t offset) :
    IntegerPackedBufferProxy<uint16_t>(ptr, scale, offset)
  { }
};

template <>
class PackedBufferProxy<int32_t> : IntegerPackedBufferProxy<int32_t> {
public:
  PackedBufferProxy<int32_t>(void* ptr, float scale, int32_t offset) :
    IntegerPackedBufferProxy<int32_t>(ptr, scale, offset)
  { }
};

template <>
class PackedBufferProxy<uint32_t> : IntegerPackedBufferProxy<uint32_t> {
public:
  PackedBufferProxy<uint32_t>(void* ptr, float scale, int32_t offset) :
    IntegerPackedBufferProxy<uint32_t>(ptr, scale, offset)
  { }
};

template <>
class PackedBufferProxy<int64_t> : IntegerPackedBufferProxy<int64_t> {
public:
  PackedBufferProxy<int64_t>(void* ptr, float scale, int32_t offset) :
    IntegerPackedBufferProxy<int64_t>(ptr, scale, offset)
  { }
};

template <>
class PackedBufferProxy<uint64_t> : IntegerPackedBufferProxy<uint64_t> {
public:
  PackedBufferProxy<uint64_t>(void* ptr, float scale, int32_t offset) :
    IntegerPackedBufferProxy<uint64_t>(ptr, scale, offset)
  { }
};

#endif /* PACKED_BUFFER_PROXY_H */
