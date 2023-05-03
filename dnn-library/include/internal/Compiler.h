/*-------------------------------------------------------------------------
 * Copyright (C) 2023, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _COMPILER_H_
#define _COMPILER_H_

#if defined(__GNUC__) && !defined(__llvm__)
#define COMPILER_GCC 1
#define COMPILER_CLANG 0
#elif defined(__clang__)
#define COMPILER_GCC 0
#define COMPILER_CLANG 1
#else
#error "Could not determine the compiler"
#endif

// Validate the compiler detection recipe:
// - all the COMPILER_XXX macros are defined
// - exactly one COMPILER_XXX macro is 1, while the others are 0.

#if not defined(COMPILER_GCC) or not defined(COMPILER_CLANG)
#error "All the COMPILER_XXX macros should be defined as 0 or 1"
#endif

#if not((COMPILER_GCC == 0 and COMPILER_CLANG == 1) or (COMPILER_GCC == 1 and COMPILER_CLANG == 0))
#error "One COMPILER_XXX must be 1 while the others are zero"
#endif

#if COMPILER_CLANG
#define FOR_CLANG(x) x
#define FOR_CLANG_COMMA ,
#define FOR_GCC(x)
#define FOR_GCC_COMMA
#elif COMPILER_GCC
#define FOR_GCC(x) x
#define FOR_GCC_COMMA ,
#define FOR_CLANG(x)
#define FOR_CLANG_COMMA
#endif

typedef unsigned int u32_t;
typedef signed int s32_t;
typedef float f32_t;
typedef void x32_t;

enum class EElementKind {
  u32, // Unsigned 32 bits integer
  s32, // Singed 32 bits integer
  f32, // Single-precision or 32 bits floating point
};

template <typename T> struct traits {
  static const bool is_vector = false;
  static const unsigned int length = 0;
  using element_type = void;
  static const bool is_floating = false;
};

template <unsigned int n, typename T> struct vtype { typedef void vector_type; };

template <unsigned int n, typename T> using v = typename vtype<n, T>::vector_type;

#if COMPILER_CLANG
typedef s32_t v8s32_t __attribute__((vector_size(sizeof(s32_t) * 8)));
template <> struct vtype<8, s32_t> { typedef v8s32_t vector_type; };
template <> struct traits<v8s32_t> {
  static const bool is_vector = true;
  static const unsigned int length = 8;
  using element_type = s32_t;
  static const bool is_floating = false;
};
#else
typedef float v8u32_t;
#endif

#if COMPILER_CLANG
typedef u32_t v8u32_t __attribute__((vector_size(sizeof(u32_t) * 8)));
template <> struct vtype<8, u32_t> { typedef v8u32_t vector_type; };
template <> struct traits<v8u32_t> {
  static const bool is_vector = true;
  static const unsigned int length = 8;
  using element_type = u32_t;
  static const bool is_floating = false;
};
#else
typedef float v8s32_t;
#endif

#if COMPILER_CLANG
typedef f32_t v8f32_t __attribute__((vector_size(sizeof(f32_t) * 8)));
template <> struct vtype<8, f32_t> { typedef v8f32_t vector_type; };
template <> struct traits<v8f32_t> {
  static const bool is_vector = true;
  static const unsigned int length = 8;
  using element_type = f32_t;
  static const bool is_floating = true;
};
#else
typedef float v8f32_t;
#endif

#endif