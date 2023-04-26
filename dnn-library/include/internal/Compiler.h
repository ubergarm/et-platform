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

#endif