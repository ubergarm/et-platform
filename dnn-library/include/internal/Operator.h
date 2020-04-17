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

#ifndef OPERATOR_H
#define OPERATOR_H

#include "LibCommon.h"
#include "utils.h"
#include "Operators.h"

#define OPERATION_STEP1   \
           "flw.ps f31, %[gatherValues]\n"               \
           "fgb.ps  f0, f31(%[src1]) \n"                 \
           "fgb.ps  f1, f31(%[src2]) \n"

#define OPERATION_STEP2   \
           "fcvt.ps.pw f1, f1 \n"                        \
           "fbc.ps f29, 0x4(%[scale]) \n"                \
           "fmul.ps f1, f1, f29 \n"                      \
           "fcvt.ps.pw f0, f0 \n"                        \
           "fbc.ps f29, 0x0(%[scale]) \n"

#define OPERATION_STEP3   \
           "fbc.ps f29, 0x8(%[scale]) \n"                \
           "frcp.ps f29, f29 \n"                         \
           "fmul.ps f0, f0, f29 \n"                      \
           "fcvt.pw.ps f0, f0 \n"

namespace dnn_lib {
  
template <typename src1Type, typename src2Type, typename dstType, typename opType> class Operator {
public:
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<!std::is_same<S, Addresser<float16>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<int8_t>>::value && !std::is_same<S, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {


  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<!std::is_same<S, Addresser<float16>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, const float *scale, const int32_t *offset) {


  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<!std::is_same<S, Addresser<float16>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr, uintptr_t dstAddr, const float *scale, const int32_t *offset) {


  }

  template <typename U = opType,
            typename std::enable_if<std::is_same<U, Add>::value,
                                    std::size_t>::type = 0>
  void doOp(dstType &dst, const src1Type &src1, const src2Type &src2, uint64_t &d,
            uint64_t &s1, uint64_t &s2) {
    dst[d] = src1[s1] + src2[s2];
  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    float  scratch0, scratch1, scratch2;
    __asm__ __volatile__("flw.ps %[op1], %[gatherValues]\n"
                         "fgh.ps  %[op0], %[op1](%[src1]) \n"
                         "fcvt.ps.f16 %[op0], %[op0] \n"
                         "fgh.ps  %[op1], %[op1](%[src2]) \n"
                         "fcvt.ps.f16 %[op1], %[op1] \n"
                         "fadd.ps %[op0], %[op0], %[op1] \n"
                         "fcvt.f16.ps %[op0], %[op0] \n"
                         "fsch.ps  %[op0], %[op1](%[dst]) \n"

                         : [op0] "=&f" (scratch0),
                           [op1] "=&f" (scratch1),
                           [op2] "=&f" (scratch2)
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         :  "memory");  //TODO: replace memory clobber with output operand if gather/scatter max offset is known
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    float scratch0, scratch1;
    __asm__ __volatile__("flw.ps  %[op0], %[src1] \n"
                         "flw.ps  %[op1], %[src2] \n"
                         "fadd.ps %[op0], %[op0], %[op1]\n"
                         "fsw.ps  %[op0], %[dst] \n"
                         : [op0] "=&f" (scratch0),
                           [op1] "=&f" (scratch1),
                           [ dst ] "=m"(*(int32_t (*)[8]) dstAddr)
                         : [ src1 ] "m"(*(const char (*)[32]) srcAddr1),
                           [ src2 ] "m"(*(const char (*)[32]) srcAddr2)
                         );
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>

  void doOpVect( int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f30, 0x4+%[offset] \n"
                         "fbc.ps f29, 0x4+%[scale] \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fsub.pi f1, f1, f30 \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fbc.ps f30, %[offset] \n"
                         "fbc.ps f29, %[scale] \n"
                         "fsub.pi f0, f0, f30 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmadd.ps f0, f0, f29, f1 \n"
                         "fbc.ps f30, 0x8+%[offset] \n"
                         "fbc.ps f29, 0x8+%[scale] \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmadd.ps f0, f0, f29, f30 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m" (*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ offset ] "m"(*(const char(*)[12]) offset),
                           [ scale ]  "m"(*(const char(*)[12]) scale)
                         : "f0", "f1", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>

  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsrli.pi f1, f0, 0x8 \n"
                         "fxor.pi f29, f29, f29 \n"
                         "fcmov.ps f0, f1, f29, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f1, f1, 0xff \n"
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n"
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f1, f1, 0xff \n"
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsrli.pi f1, f0, 0x8 \n"
                         "fxor.pi f29, f29, f29 \n"
                         "fcmov.ps f0, f1, f29, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n"
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsrli.pi f1, f0, 0x8 \n"
                         "fxor.pi f29, f29, f29 \n"
                         "fcmov.ps f0, f1, f29, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect( int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n"
                         "fandi.pi f1, f1, 0xff \n"
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect( int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n"
                         "fandi.pi f1, f1, 0xff \n"
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsrli.pi f1, f0, 0x8 \n"
                         "fxor.pi f29, f29, f29 \n"
                         "fcmov.ps f0, f1, f29, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect( int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fgh.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "fgh.ps  f1, f31(%[src2]) \n"
                         "fcvt.ps.f16 f1, f1 \n"
                         "fsub.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect( int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps  f0, 0x0(%[src1]) \n"
                         "flw.ps  f1, 0x0(%[src2]) \n"
                         "fsub.ps f0, f0, f1 \n"
                         "fsw.ps  f0, 0x0(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>

  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f30, 0x4(%[offset]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fsub.pi f1, f1, f30 \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fbc.ps f30, 0x0(%[offset]) \n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fsub.pi f0, f0, f30 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmsub.ps f0, f0, f29, f1 \n"
                         "fbc.ps f30, 0x8(%[offset]) \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmadd.ps f0, f0, f29, f30 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>

  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsrli.pi f1, f0, 0x8 \n"
                         "fxor.pi f29, f29, f29 \n"
                         "fcmov.ps f0, f1, f29, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f1, f1, 0xff \n"
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n"
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f1, f1, 0xff \n"
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsrli.pi f1, f0, 0x8 \n"
                         "fxor.pi f29, f29, f29 \n"
                         "fcmov.ps f0, f1, f29, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n"
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsrli.pi f1, f0, 0x8 \n"
                         "fxor.pi f29, f29, f29 \n"
                         "fcmov.ps f0, f1, f29, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n"
                         "fandi.pi f1, f1, 0xff \n"
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n"
                         "fandi.pi f1, f1, 0xff \n"
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n"
                         OPERATION_STEP3
                         "fsrli.pi f1, f0, 0x8 \n"
                         "fxor.pi f29, f29, f29 \n"
                         "fcmov.ps f0, f1, f29, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType,
            typename std::enable_if<std::is_same<U, Sub>::value,
                                    std::size_t>::type = 0>
  void doOp(dstType &dst, const src1Type &src1, const src2Type &src2, uint64_t &d,
            uint64_t &s1, uint64_t &s2) {
    dst[d] = src1[s1] - src2[s2];
  }

   template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Mul>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fgh.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "fgh.ps  f1, f31(%[src2]) \n"
                         "fcvt.ps.f16 f1, f1 \n"
                         "fmul.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Mul>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps  f0, 0x0(%[src1]) \n"
                         "flw.ps  f1, 0x0(%[src2]) \n"
                         "fmul.ps f0, f0, f1 \n"
                         "fsw.ps  f0, 0x0(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "memory");
  }

  template <typename U = opType, typename S = src1Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Mul>::value && std::is_same<D, Addresser<uint8_t>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fandi.pi f0, f0, 0xff \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fandi.pi f1, f1, 0xff \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fmul.ps f0, f1, f0 \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsatu8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Mul>::value && (std::is_same<S1, Addresser<uint8_t>>::value || std::is_same<S2, Addresser<uint8_t>>::value) && std::is_same<D, Addresser<int8_t>>::value && !std::is_same<S1, Addresser<float>>::value && !std::is_same<S1, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fmul.ps f0, f1, f0 \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Mul>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value, std::size_t>::type = 0>

  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f30, 0x0(%[offset]) \n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fsub.pi f0, f0, f30 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fbc.ps f30, 0x4(%[offset]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fsub.pi f1, f1, f30 \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fmul.ps f0, f0, f1 \n"
                         "fbc.ps f30, 0x8(%[offset]) \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmadd.ps f0, f0, f29, f30 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType,
            typename std::enable_if<std::is_same<U, Mul>::value,
                                    std::size_t>::type = 0>
  void doOp(dstType &dst, const src1Type &src1, const src2Type &src2, uint64_t &d,
            uint64_t &s1, uint64_t &s2) {
    dst[d] = src1[s1] * src2[s2];
  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Div>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fgh.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "fgh.ps  f1, f31(%[src2]) \n"
                         "fcvt.ps.f16 f1, f1 \n"
                         "frcp.ps f1, f1 \n"
                         "fmul.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Div>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps  f0, 0x0(%[src1]) \n"
                         "flw.ps  f1, 0x0(%[src2]) \n"
                         "frcp.ps f1, f1 \n"
                         "fmul.ps f0, f0, f1 \n"
                         "fsw.ps  f0, 0x0(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "memory");
  }

  template <typename U = opType, typename S = src1Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Div>::value && std::is_same<D, Addresser<uint8_t>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fandi.pi f0, f0, 0xff \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fandi.pi f1, f1, 0xff \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "frcp.ps f1, f1 \n"
                         "fmul.ps f0, f1, f0 \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsatu8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Div>::value && (std::is_same<S1, Addresser<uint8_t>>::value || std::is_same<S2, Addresser<uint8_t>>::value) && std::is_same<D, Addresser<int8_t>>::value && !std::is_same<S1, Addresser<float>>::value && !std::is_same<S1, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "frcp.ps f1, f1 \n"
                         "fmul.ps f0, f1, f0 \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m" (*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Div>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value, std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n"
                         "flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f30, 0x0(%[offset]) \n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fsub.pi f0, f0, f30 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fbc.ps f30, 0x4(%[offset]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fsub.pi f1, f1, f30 \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "frcp.ps f1, f1 \n"
                         "fmul.ps f0, f0, f1 \n"
                         "fbc.ps f30, 0x8(%[offset]) \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmadd.ps f0, f0, f29, f30 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f30", "f31", "memory");
  }

  template <
      typename U = opType, typename S = src1Type,
      typename std::enable_if<std::is_same<U, Div>::value &&
                                  std::is_same<S, Addresser<int64_t>>::value,
                              std::size_t>::type = 0>
  void doOp(S &dst, const S &src1, const S &src2, uint64_t &d, uint64_t &s1,
            uint64_t &s2) {
    float inverted_op, tmp_res;
    // TODO Do proper conversion int64 to float
    int32_t tmp2 = (int32_t)src2[s2];
    int32_t tmp1 = (int32_t)src1[s1];
    int32_t res;
    getReciprocal(float(tmp2), inverted_op);
    tmp_res = float(tmp1) * inverted_op;
    res = (int32_t)tmp_res;
    dst[d] = (int64_t)res;
  }

  template <
      typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
      typename std::enable_if<std::is_same<U, Div>::value &&
                                  !std::is_same<S1, Addresser<int64_t>>::value,
                              std::size_t>::type = 0>
  void doOp(D &dst, const S1 &src1, const S2 &src2, uint64_t &d, uint64_t &s1,
            uint64_t &s2) {
    float inverted_op;
    getReciprocal(src2[s2], inverted_op);
    dst[d] = src1[s1] * inverted_op;
  }

   template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Max>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fgh.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "fgh.ps  f1, f31(%[src2]) \n"
                         "fcvt.ps.f16 f1, f1 \n"
                         "fmax.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Max>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps  f0, 0x0(%[src1]) \n"
                         "flw.ps  f1, 0x0(%[src2]) \n"
                         "fmax.ps f0, f0, f1 \n"
                         "fsw.ps  f0, 0x0(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "memory");
  }

  template <typename U = opType, typename S = src1Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Max>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<float16>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fandi.pi f0, f0, 0xff \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fandi.pi f1, f1, 0xff \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fmax.ps f0, f1, f0 \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsatu8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Max>::value && (std::is_same<S1, Addresser<uint8_t>>::value || std::is_same<S2, Addresser<uint8_t>>::value) && std::is_same<D, Addresser<int8_t>>::value && !std::is_same<S1, Addresser<float>>::value && !std::is_same<S1, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fmax.ps f0, f1, f0 \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Max>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value, std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f30, 0x0(%[offset]) \n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fsub.pi f0, f0, f30 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fbc.ps f30, 0x4(%[offset]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fsub.pi f1, f1, f30 \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fmax.ps f0, f0, f1 \n"
                         "fbc.ps f30, 0x8(%[offset]) \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmadd.ps f0, f0, f29, f30 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType,
            typename std::enable_if<std::is_same<U, Max>::value,
                                    std::size_t>::type = 0>
  void doOp(dstType &dst, const src1Type &src1, const src2Type &src2, uint64_t &d,
            uint64_t &s1, uint64_t &s2) {
    dst[d] = std::max(src1[s1], src2[s2]);
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Min>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fgh.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "fgh.ps  f1, f31(%[src2]) \n"
                         "fcvt.ps.f16 f1, f1 \n"
                         "fmin.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Min>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps  f0, 0x0(%[src1]) \n"
                         "flw.ps  f1, 0x0(%[src2]) \n"
                         "fmin.ps f0, f0, f1 \n"
                         "fsw.ps  f0, 0x0(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "memory");
  }

  template <typename U = opType, typename S = src1Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Min>::value && std::is_same<D, Addresser<uint8_t>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fandi.pi f0, f0, 0xff \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmin.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fandi.pi f1, f1, 0xff \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fmul.ps f0, f1, f0 \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsatu8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Min>::value && (std::is_same<S1, Addresser<uint8_t>>::value || std::is_same<S2, Addresser<uint8_t>>::value) && std::is_same<D, Addresser<int8_t>>::value && !std::is_same<S1, Addresser<float>>::value && !std::is_same<S1, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fmin.ps f0, f1, f0 \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Min>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value, std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f30, 0x0(%[offset]) \n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fsub.pi f0, f0, f30 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fbc.ps f30, 0x4(%[offset]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fsub.pi f1, f1, f30 \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fmin.ps f0, f0, f1 \n"
                         "fbc.ps f30, 0x8(%[offset]) \n"
                         "fbc.ps f29, 0x8(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmadd.ps f0, f0, f29, f30 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType,
            typename std::enable_if<std::is_same<U, Min>::value,
                                    std::size_t>::type = 0>
  void doOp(dstType &dst, const src1Type &src1, const src2Type &src2, uint64_t &d,
            uint64_t &s1, uint64_t &s2) {
    dst[d] = std::min(src1[s1], src2[s2]);
  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpEQ>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, const float *scale, const int32_t *offset) {
    uint32_t scatterValues[]={0,1,2,3,4,5,6,7};
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fgh.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "fgh.ps  f1, f31(%[src2]) \n"
                         "flw.ps f31, %[scatterValues]\n"
                         "fcvt.ps.f16 f1, f1 \n"
                         "feq.ps f0, f0, f1 \n"
                         "fandi.pi f0, f0, 0x1 \n"
                         "fscb.ps f0, f31(%[dst])\n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ scatterValues ] "m"(*(const int32_t (*)[8]) scatterValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpEQ>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps  f0, 0x0(%[src1]) \n"
                         "flw.ps  f1, 0x0(%[src2]) \n"
                         "feq.ps f0, f0, f1 \n"
                         "fandi.pi f0, f0, 0x1 \n"
                         "fslli.pi f0, f0, 24 \n"
                         "fbci.pi f2, 0 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsw f2, 0x0(%[dst]) \n"
                         "mov.m.x m1, zero, 16 \n"
                         "maskand m0, m0, m1 \n"
                         "fsw.ps f2, -12(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f2", "memory");

  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpEQ>::value && std::is_same<S, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f30, 0x0(%[offset]) \n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fsub.pi f0, f0, f30 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fbc.ps f30, 0x4(%[offset]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fsub.pi f1, f1, f30 \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "feq.ps f0, f0, f1 \n"
                         "fandi.pi f0, f0, 0x1 \n"
                         "fslli.pi f0, f0, 24 \n"
                         "fbci.pi f2, 0 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsw f2, 0x0(%[dst]) \n"
                         "mov.m.x m1, zero, 16 \n"
                         "maskand m0, m0, m1 \n"
                         "fsw.ps f2, -12(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f2", "f29", "f30", "f31", "memory");
  }


  template <typename U = opType,
            typename std::enable_if<std::is_same<U, CmpEQ>::value,
                                    std::size_t>::type = 0>
  void doOp(bool *dst, const src1Type &src1, const src2Type &src2, uint64_t &d,
            uint64_t &s1, uint64_t &s2) {
    dst[d] = (src1[s1] == src2[s2]) ? true : false;
  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpLTE>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, const float *scale, const int32_t *offset) {
    uint32_t scatterValues[]={0,1,2,3,4,5,6,7};
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fgh.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "fgh.ps  f1, f31(%[src2]) \n"
                         "flw.ps f31, %[scatterValues]\n"
                         "fcvt.ps.f16 f1, f1 \n"
                         "fle.ps f0, f0, f1 \n"
                         "fandi.pi f0, f0, 0x1 \n"
                         "fscb.ps f0, f31(%[dst])\n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ scatterValues ] "m"(*(const int32_t (*)[8]) scatterValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
    
  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpLTE>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps  f0, 0x0(%[src1]) \n"
                         "flw.ps  f1, 0x0(%[src2]) \n"
                         "fle.ps f0, f0, f1 \n"
                         "fandi.pi f0, f0, 0x1 \n"
                         "fslli.pi f0, f0, 24 \n"
                         "fbci.pi f2, 0 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsw f2, 0x0(%[dst]) \n"
                         "mov.m.x m1, zero, 16 \n"
                         "maskand m0, m0, m1 \n"
                         "fsw.ps f2, -12(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f2", "memory");

  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpLTE>::value && std::is_same<S, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f30, 0x0(%[offset]) \n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fsub.pi f0, f0, f30 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fbc.ps f30, 0x4(%[offset]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fsub.pi f1, f1, f30 \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "fle.ps f0, f0, f1 \n"
                         "fandi.pi f0, f0, 0x1 \n"
                         "fslli.pi f0, f0, 24 \n"
                         "fbci.pi f2, 0 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsw f2, 0x0(%[dst]) \n"
                         "mov.m.x m1, zero, 16 \n"
                         "maskand m0, m0, m1 \n"
                         "fsw.ps f2, -12(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f2", "f29", "f30", "f31", "memory");

  }

  template <typename U = opType,
            typename std::enable_if<std::is_same<U, CmpLTE>::value,
                                    std::size_t>::type = 0>
  void doOp(bool *dst, const src1Type &src1, const src2Type &src2, uint64_t &d,
            uint64_t &s1, uint64_t &s2) {
    dst[d] = (src1[s1] <= src2[s2]) ? true : false;
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpLT>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, const float *scale, const int32_t *offset) {
    uint32_t scatterValues[]={0,1,2,3,4,5,6,7};
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fgh.ps  f0, f31(%[src1]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "fgh.ps  f1, f31(%[src2]) \n"
                         "flw.ps f31, %[scatterValues]\n"
                         "fcvt.ps.f16 f1, f1 \n"
                         "flt.ps f0, f0, f1 \n"
                         "fandi.pi f0, f0, 0x1 \n"
                         "fscb.ps f0, f31(%[dst])\n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ scatterValues ] "m"(*(const int32_t (*)[8]) scatterValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpLT>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps  f0, 0x0(%[src1]) \n"
                         "flw.ps  f1, 0x0(%[src2]) \n"
                         "flt.ps f0, f0, f1 \n"
                         "fandi.pi f0, f0, 0x1 \n"
                         "fslli.pi f0, f0, 24 \n"
                         "fbci.pi f2, 0 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsw f2, 0x0(%[dst]) \n"
                         "mov.m.x m1, zero, 16 \n"
                         "maskand m0, m0, m1 \n"
                         "fsw.ps f2, -12(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f2", "memory");

  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpLT>::value && std::is_same<S, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, const float *scale, const int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f30, 0x0(%[offset]) \n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src1]) \n"
                         "fsub.pi f0, f0, f30 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "fgb.ps  f1, f31(%[src2]) \n"
                         "fbc.ps f30, 0x4(%[offset]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "fsub.pi f1, f1, f30 \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"
                         "flt.ps f0, f0, f1 \n"
                         "fandi.pi f0, f0, 0x1 \n"
                         "fslli.pi f0, f0, 24 \n"
                         "fbci.pi f2, 0 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsrli.pi f2, f2, 0x8 \n"
                         "fswizz.ps f0, f0, 0x39 \n"
                         "for.pi f2, f0, f2 \n"
                         "fsw f2, 0x0(%[dst]) \n"
                         "mov.m.x m1, zero, 16 \n"
                         "maskand m0, m0, m1 \n"
                         "fsw.ps f2, -12(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f2", "f29", "f30", "f31", "memory");

  }

  template <typename U = opType,
            typename std::enable_if<std::is_same<U, CmpLT>::value,
                                    std::size_t>::type = 0>
  void doOp(bool *dst, const src1Type &src1, const src2Type &src2, uint64_t &d,
            uint64_t &s1, uint64_t &s2) {
    dst[d] = (src1[s1] < src2[s2]) ? true : false;
  }

  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Pow>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    float half = 0.5;
    float minus2 = -2;
    __asm__ __volatile__("maskand m1, m0, m0 \n"
                         "flw.ps f28, %[gatherValues]\n"
                         "fgh.ps  f0, f28(%[src1]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "fgh.ps  f1, f28(%[src2]) \n"
                         "fcvt.ps.f16 f1, f1 \n"

                         "fbc.ps f31, 0x0(%[half]) \n"
                         "fbc.ps f30, 0x0(%[minus2]) \n"
                         "fxor.pi f29, f29, f29 \n"
                         "feqm.ps m0, f29, f0 \n"
                         "feq.pi f2, f1, f29 \n"
                         "fandi.pi f2, f2, 0x1 \n"
                         "fcvt.ps.pw f2, f2 \n"
                         "masknot m0, m0 \n"
                         "maskand m0, m0, m1 \n"
                         "fmul.ps f3, f0, f0 \n"  //f3 has a*a
                         "flog.ps f2, f3 \n" //f2 has log_2(a^2)
                         "fmul.ps f4, f31, f1 \n" //f4 has b/2
                         "fmul.ps f2, f2, f4 \n" //f2 has log_2(a^2) * b/2
                         "fexp.ps f2, f2 \n"
                         "flem.ps m2, f0, f29 \n"

                         "maskand m0, m2, m0  \n"
                         "fround.ps f5, f1 \n" //f5 has rounded b
                         "maskxor m3, m3, m3\n"
                         "feqm.ps m3, f5, f1 \n" //m3 has 1 if b is integer 0 if not
                         "masknot m4, m3 \n"
                         "maskand m0, m0, m4 \n"
                         "fnot.pi f2, f29 \n"
                         "maskand m0, m3, m2 \n"

                         "fcvt.pw.ps f5, f5\n"
                         "fandi.pi f5, f5, 0x1\n"
                         "fcvt.pw.ps f30, f30\n"
                         "fmul.pi f5, f5, f30\n"
                         "faddi.pi f5, f5, 0x1\n"
                         "fcvt.ps.pw f5, f5 \n"
                         "fmul.ps f2, f2, f5 \n"
                         "maskand m0, m1, m1 \n"

                         "fcvt.f16.ps f2, f2 \n"
                         "fsch.ps  f2, f28(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr),
                           [ half ] "r"(&half),
                           [ minus2 ] "r"(&minus2),
                           [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues)
                         : "f0", "f1", "f2", "f3", "f4", "f5", "f28", "f29", "f30", "f31", "memory");
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Pow>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    float half = 0.5;
    float minus2 = -2;
    __asm__ __volatile__("maskand m1, m0, m0 \n"
                         "flw.ps  f0, 0x0(%[src1]) \n"  //f0 has a
                         "flw.ps  f1, 0x0(%[src2]) \n"  //f1 has b
                         "fbc.ps f31, 0x0(%[half]) \n"
                         "fbc.ps f30, 0x0(%[minus2]) \n"
                         "fxor.pi f29, f29, f29 \n"
                         "feqm.ps m0, f29, f0 \n"
                         "feq.pi f2, f1, f29 \n"
                         "fandi.pi f2, f2, 0x1 \n"
                         "fcvt.ps.pw f2, f2 \n"
                         "masknot m0, m0 \n"
                         "maskand m0, m0, m1 \n"
                         "fmul.ps f3, f0, f0 \n"  //f3 has a*a
                         "flog.ps f2, f3 \n" //f2 has log_2(a^2)
                         "fmul.ps f4, f31, f1 \n" //f4 has b/2
                         "fmul.ps f2, f2, f4 \n" //f2 has log_2(a^2) * b/2
                         "fexp.ps f2, f2 \n"
                         "flem.ps m2, f0, f29 \n"

                         "maskand m0, m2, m0  \n"
                         "fround.ps f5, f1 \n" //f5 has rounded b
                         "maskxor m3, m3, m3\n"
                         "feqm.ps m3, f5, f1 \n" //m3 has 1 if b is integer 0 if not
                         "masknot m4, m3 \n"
                         "maskand m0, m0, m4 \n"
                         "fnot.pi f2, f29 \n"
                         "maskand m0, m3, m2 \n"

                         "fcvt.pw.ps f5, f5\n"
                         "fandi.pi f5, f5, 0x1\n"
                         "fcvt.pw.ps f30, f30\n"
                         "fmul.pi f5, f5, f30\n"
                         "faddi.pi f5, f5, 0x1\n"
                         "fcvt.ps.pw f5, f5 \n"
                         "fmul.ps f2, f2, f5 \n"
                         "maskand m0, m1, m1 \n"
                         "fsw.ps  f2, 0x0(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr),
                           [ half ] "r"(&half),
                           [ minus2 ] "r"(&minus2)
                         : "f0", "f1", "f2", "f3", "f4", "f5", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Pow>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value, std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    float half = 0.5;
    float minus2 = -2;
    __asm__ __volatile__("maskand m1, m0, m0 \n"
                         "fbc.ps f31, 0x0(%[half]) \n"
                         "fbc.ps f30, 0x0(%[minus2]) \n"
                         "flw.ps f28, %[gatherValues]\n"
                         "fbc.ps f26, 0x0(%[offset]) \n"
                         "fbc.ps f27, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f28(%[src1]) \n"
                         "fsub.pi f0, f0, f26 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f27 \n"
                         "fgb.ps  f1, f28(%[src2]) \n"
                         "fbc.ps f26, 0x4(%[offset]) \n"
                         "fbc.ps f27, 0x4(%[scale]) \n"
                         "fsub.pi f1, f1, f26 \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"

                         "fxor.pi f29, f29, f29 \n"
                         "feqm.ps m0, f29, f0 \n"
                         "feq.pi f2, f1, f29 \n"
                         "fandi.pi f2, f2, 0x1 \n"
                         "fcvt.ps.pw f2, f2 \n"
                         "masknot m0, m0 \n"
                         "maskand m0, m0, m1 \n"
                         "fmul.ps f3, f0, f0 \n"  //f3 has a*a
                         "flog.ps f2, f3 \n" //f2 has log_2(a^2)
                         "fmul.ps f4, f31, f1 \n" //f4 has b/2
                         "fmul.ps f2, f2, f4 \n" //f2 has log_2(a^2) * b/2
                         "fexp.ps f2, f2 \n"
                         "flem.ps m2, f0, f29 \n"

                         "maskand m0, m2, m0  \n"
                         "fround.ps f5, f1 \n" //f5 has rounded b
                         "maskxor m3, m3, m3\n"
                         "feqm.ps m3, f5, f1 \n" //m3 has 1 if b is integer 0 if not
                         "masknot m4, m3 \n"
                         "maskand m0, m0, m4 \n"
                         "fnot.pi f2, f29 \n"
                         "maskand m0, m3, m2 \n"

                         "fcvt.pw.ps f5, f5\n"
                         "fandi.pi f5, f5, 0x1\n"
                         "fcvt.pw.ps f30, f30\n"
                         "fmul.pi f5, f5, f30\n"
                         "faddi.pi f5, f5, 0x1\n"
                         "fcvt.ps.pw f5, f5 \n"
                         "fmul.ps f2, f2, f5 \n"
                         "maskand m0, m1, m1 \n"

                         "fbc.ps f26, 0x8(%[offset]) \n"
                         "fbc.ps f27, 0x8(%[scale]) \n"
                         "frcp.ps f29, f27 \n"
                         "fcvt.ps.pw f30, f26 \n"
                         "fmadd.ps f2, f2, f29, f30 \n"
                         "fcvt.pw.ps f2, f2 \n"
                         "fsat8.pi f2, f2 \n"
                         "fscb.ps  f2, f28(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr),
                           [ half ] "r"(&half),
                           [ minus2 ] "r"(&minus2),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale),
                           [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues)
                         : "f0", "f1", "f2", "f3", "f4", "f5", "f26", "f27", "f28", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Pow>::value && (std::is_same<S1, Addresser<uint8_t>>::value || std::is_same<S2, Addresser<uint8_t>>::value) && std::is_same<D, Addresser<int8_t>>::value && !std::is_same<S1, Addresser<float>>::value && !std::is_same<S1, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    float half = 0.5;
    float minus2 = -2;
    __asm__ __volatile__("maskand m1, m0, m0 \n"
                         "fbc.ps f31, 0x0(%[half]) \n"
                         "fbc.ps f30, 0x0(%[minus2]) \n"
                         "flw.ps f28, %[gatherValues]\n"
                         "fbc.ps f27, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f28(%[src1]) \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f27 \n"
                         "fgb.ps  f1, f28(%[src2]) \n"
                         "fbc.ps f27, 0x4(%[scale]) \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"

                         "fxor.pi f29, f29, f29 \n"
                         "feqm.ps m0, f29, f0 \n"
                         "feq.pi f2, f1, f29 \n"
                         "fandi.pi f2, f2, 0x1 \n"
                         "fcvt.ps.pw f2, f2 \n"
                         "masknot m0, m0 \n"
                         "maskand m0, m0, m1 \n"
                         "fmul.ps f3, f0, f0 \n"  //f3 has a*a
                         "flog.ps f2, f3 \n" //f2 has log_2(a^2)
                         "fmul.ps f4, f31, f1 \n" //f4 has b/2
                         "fmul.ps f2, f2, f4 \n" //f2 has log_2(a^2) * b/2
                         "fexp.ps f2, f2 \n"
                         "flem.ps m2, f0, f29 \n"

                         "maskand m0, m2, m0  \n"
                         "fround.ps f5, f1 \n" //f5 has rounded b
                         "maskxor m3, m3, m3\n"
                         "feqm.ps m3, f5, f1 \n" //m3 has 1 if b is integer 0 if not
                         "masknot m4, m3 \n"
                         "maskand m0, m0, m4 \n"
                         "fnot.pi f2, f29 \n"
                         "maskand m0, m3, m2 \n"

                         "fcvt.pw.ps f5, f5\n"
                         "fandi.pi f5, f5, 0x1\n"
                         "fcvt.pw.ps f30, f30\n"
                         "fmul.pi f5, f5, f30\n"
                         "faddi.pi f5, f5, 0x1\n"
                         "fcvt.ps.pw f5, f5 \n"
                         "fmul.ps f2, f2, f5 \n"
                         "maskand m0, m1, m1 \n"

                         "fbc.ps f27, 0x8(%[scale]) \n"
                         "frcp.ps f29, f27 \n"
                         "fmul.ps f2, f2, f29 \n"
                         "fcvt.pw.ps f2, f2 \n"
                         "fsat8.pi f2, f2 \n"
                         "fscb.ps  f2, f28(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr),
                           [ half ] "r"(&half),
                           [ minus2 ] "r"(&minus2),
                           [ scale ] "r"(scale),
                           [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues)
                         : "f0", "f1", "f2", "f3", "f4", "f5", "f27", "f28", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType, typename S = src1Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Pow>::value && std::is_same<D, Addresser<uint8_t>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    float half = 0.5;
    float minus2 = -2;
    __asm__ __volatile__("maskand m1, m0, m0 \n"
                         "fbc.ps f31, 0x0(%[half]) \n"
                         "fbc.ps f30, 0x0(%[minus2]) \n"
                         "flw.ps f28, %[gatherValues]\n"
                         "fbc.ps f27, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f28(%[src1]) \n"
                         "fandi.pi f0, f0, 0xff \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f27 \n"
                         "fgb.ps  f1, f28(%[src2]) \n"
                         "fandi.pi f1, f1, 0xff \n"
                         "fbc.ps f27, 0x4(%[scale]) \n"
                         "fcvt.ps.pw f1, f1 \n"
                         "fmul.ps f1, f1, f29 \n"

                         "fxor.pi f29, f29, f29 \n"
                         "feqm.ps m0, f29, f0 \n"
                         "feq.pi f2, f1, f29 \n"
                         "fandi.pi f2, f2, 0x1 \n"
                         "fcvt.ps.pw f2, f2 \n"
                         "masknot m0, m0 \n"
                         "maskand m0, m0, m1 \n"
                         "fmul.ps f3, f0, f0 \n"  //f3 has a*a
                         "flog.ps f2, f3 \n" //f2 has log_2(a^2)
                         "fmul.ps f4, f31, f1 \n" //f4 has b/2
                         "fmul.ps f2, f2, f4 \n" //f2 has log_2(a^2) * b/2
                         "fexp.ps f2, f2 \n"
                         "flem.ps m2, f0, f29 \n"

                         "maskand m0, m2, m0  \n"
                         "fround.ps f5, f1 \n" //f5 has rounded b
                         "maskxor m3, m3, m3\n"
                         "feqm.ps m3, f5, f1 \n" //m3 has 1 if b is integer 0 if not
                         "masknot m4, m3 \n"
                         "maskand m0, m0, m4 \n"
                         "fnot.pi f2, f29 \n"
                         "maskand m0, m3, m2 \n"

                         "fcvt.pw.ps f5, f5\n"
                         "fandi.pi f5, f5, 0x1\n"
                         "fcvt.pw.ps f30, f30\n"
                         "fmul.pi f5, f5, f30\n"
                         "faddi.pi f5, f5, 0x1\n"
                         "fcvt.ps.pw f5, f5 \n"
                         "fmul.ps f2, f2, f5 \n"
                         "maskand m0, m1, m1 \n"

                         "fbc.ps f27, 0x8(%[scale]) \n"
                         "frcp.ps f29, f27 \n"
                         "fmul.ps f2, f2, f29 \n"
                         "fcvt.pw.ps f2, f2 \n"
                         "fsatu8.pi f2, f2 \n"
                         "fscb.ps  f2, f28(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr),
                           [ half ] "r"(&half),
                           [ minus2 ] "r"(&minus2),
                           [ scale ] "r"(scale),
                           [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues)
                         : "f0", "f1", "f2", "f3", "f4", "f5", "f27", "f28", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType,
            typename std::enable_if<std::is_same<U, Pow>::value,
                                    std::size_t>::type = 0>
  void doOp(dstType &dst, const src1Type &src1, const src2Type &src2, uint64_t &d,
            uint64_t &s1, uint64_t &s2) {
    dst[d] = getPow(src1[s1], src2[s2]);
  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, ElementLog>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    float log2e = M_1_LOG2E;
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f30, 0x0(%[log2e]) \n"
                         "fgh.ps  f0, f31(%[src]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "flog.ps f0, f0 \n"
                         "fmul.ps f0, f0, f30 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src ] "r"(srcAddr),
                           [ dst ] "r"(dstAddr),
                           [ log2e ] "r"(&log2e)
                         : "f0", "f30", "f31", "memory");
  }


  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, ElementLog>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    float log2e = M_1_LOG2E;
    __asm__ __volatile__("flw.ps  f0, 0x0(%[src]) \n"
                         "fbc.ps f30, 0x0(%[log2e]) \n"
                         "flog.ps f0, f0 \n"
                         "fmul.ps f0, f0, f30 \n"
                         "fsw.ps  f0, 0x0(%[dst]) \n"
                         :
                         : [ src ] "r"(srcAddr),
                           [ dst ] "r" (dstAddr),
                           [ log2e ] "r"(&log2e)
                         : "f0", "f30", "memory");

  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, ElementLog>::value && std::is_same<S, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(int32_t *gatherValues, uintptr_t srcAddr, uintptr_t dstAddr, const float *scale, const int32_t *offset) {
    float log2e = M_1_LOG2E;
    __asm__ __volatile__("flw.ps f31, %[gatherValues]\n"
                         "fbc.ps f28, 0x0(%[log2e]) \n"
                         "fbc.ps f30, 0x0(%[offset]) \n"
                         "fbc.ps f29, 0x0(%[scale]) \n"
                         "fgb.ps  f0, f31(%[src]) \n"
                         "fsub.pi f0, f0, f30 \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fmul.ps f0, f0, f29 \n"
                         "flog.ps f0, f0 \n"
                         "fmul.ps f0, f0, f28 \n"
                         "fbc.ps f30, 0x4(%[offset]) \n"
                         "fbc.ps f29, 0x4(%[scale]) \n"
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmadd.ps f0, f0, f29, f30 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "m"(*(const int32_t (*)[8]) gatherValues),
                           [ src ] "r"(srcAddr),
                           [ dst ] "r"(dstAddr),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale),
                           [ log2e ] "r"(&log2e)
                         : "f0", "f28", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType,
            typename std::enable_if<std::is_same<U, ElementLog>::value,
                                    std::size_t>::type = 0>
  void doOp(dstType &dst, const src1Type &src1, uint64_t &d, uint64_t &s1) {
    float op1;
    float op2 = M_1_LOG2E;
    fpLog2SingleElement(src1[s1], op1);
    dst[d] = op1 * op2;
  }

  // And Immediate version (src, imm)
  template <typename U = opType,
            typename std::enable_if<std::is_same<U, And>::value,
                                    std::size_t>::type = 0>
  void doOp(dstType *dst, const src1Type *src1, src2Type src2, uint64_t &d,
            uint64_t &s1) {
    dst[d] = src1[s1] & src2;
  }

  // Or Immediate version (src, imm)
  template <typename U = opType,
            typename std::enable_if<std::is_same<U, Or>::value,
                                    std::size_t>::type = 0>
  void doOp(dstType *dst, const src1Type *src1, src2Type src2, uint64_t &d,
            uint64_t &s1) {
    dst[d] = src1[s1] | src2;
  }

  // Xor Immediate version (src, imm)
  template <typename U = opType,
            typename std::enable_if<std::is_same<U, Xor>::value,
                                    std::size_t>::type = 0>
  void doOp(dstType *dst, const src1Type *src1, src2Type src2, uint64_t &d,
            uint64_t &s1) {
    dst[d] = src1[s1] ^ src2;
  }
};

} //namespace dnn_lib
#endif /* OPERATOR_H */
