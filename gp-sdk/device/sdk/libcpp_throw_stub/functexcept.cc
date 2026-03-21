/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#include <etsoc/common/utils.h>

/* stub implementations of __throw__XXX libstdc++ functions.
 * as we are working with -nostdlib -fno-exceptions.. we need a way for the stdlib headers
 * to abort. (This is tipically delegated to libsdc++ and it decides based on having exceptions
 * enabled or not).
 *
 * based on https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/src/c%2B%2B11/functexcept.cc
 */

namespace std {
// declarations
//
void __throw_bad_exception();
void __throw_bad_alloc();
void __throw_bad_array_new_length();
void __throw_bad_cast();
void __throw_bad_typeid();
void __throw_logic_error(const char* __s __attribute__((unused)));
void __throw_domain_error(const char* __s __attribute__((unused)));
void __throw_invalid_argument(const char* __s __attribute__((unused)));
void __throw_length_error(const char* __s __attribute__((unused)));
void __throw_out_of_range(const char* __s __attribute__((unused)));
void __throw_out_of_range_fmt(const char* __fmt, ...);
void __throw_runtime_error(const char* __s __attribute__((unused)));
void __throw_range_error(const char* __s __attribute__((unused)));
void __throw_overflow_error(const char* __s __attribute__((unused)));
void __throw_underflow_error(const char* __s __attribute__((unused)));

// definitions

void __attribute__((weak)) __throw_bad_exception() {
  et_assert(false);
  et_abort();
}

void __attribute__((weak)) __throw_bad_alloc() {
  et_assert(false);
  et_abort();
}

void __attribute__((weak)) __throw_bad_array_new_length() {
  et_assert(false);
  et_abort();
}

void __attribute__((weak)) __throw_bad_cast() {
  et_assert(false);
  et_abort();
}

void __attribute__((weak)) __throw_bad_typeid() {
  et_assert(false);
  et_abort();
}

void __attribute__((weak)) __throw_logic_error(const char* __s __attribute__((unused))) {
  et_assert(false && __s);
  et_abort();
}

void __attribute__((weak)) __throw_domain_error(const char* __s __attribute__((unused))) {
  et_assert(false && __s);
  et_abort();
}

void __attribute__((weak)) __throw_invalid_argument(const char* __s __attribute__((unused))) {
  et_assert(false && __s);
  et_abort();
}

void __attribute__((weak)) __throw_length_error(const char* __s __attribute__((unused))) {
  et_assert(false && __s);
  et_abort();
}

void __attribute__((weak)) __throw_out_of_range(const char* __s __attribute__((unused))) {
  et_assert(false && __s);
  et_abort();
}

void __attribute__((weak)) __throw_out_of_range_fmt(const char* __fmt, ...) {
  __throw_out_of_range(__fmt);
}

void __attribute__((weak)) __throw_runtime_error(const char* __s __attribute__((unused))) {
  et_assert(false && __s);
  et_abort();
}

void __attribute__((weak)) __throw_range_error(const char* __s __attribute__((unused))) {
  et_assert(false && __s);
  et_abort();
}

void __attribute__((weak)) __throw_overflow_error(const char* __s __attribute__((unused))) {
  et_assert(false && __s);
  et_abort();
}

void __attribute__((weak)) __throw_underflow_error(const char* __s __attribute__((unused))) {
  et_assert(false && __s);
  et_abort();
}

} // namespace std
