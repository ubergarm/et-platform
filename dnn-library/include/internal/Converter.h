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

#ifndef CONVERTER_H
#define CONVERTER_H

template <typename SRC, typename DST> class Converter {
public:
  Converter(){};

  // TODO Do proper conversion (currently we convert fp16 and int64 through
  // int32)
  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, float>::value &&
                                        std::is_same<D, int64_t>::value,
                                    std::size_t>::type = 0>
  DST convert(SRC s) {
    int32_t tmp = (int32_t)s;
    return (DST)tmp;
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, float>::value &&
                                        std::is_same<D, int64_t>::value,
                                    std::size_t>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *,  int32_t *) {
    const int32_t gatherValues1[] __attribute__((aligned(32))) = {0, 0, 4, 4, 8, 8, 12, 12};
    float scratch0, scratch1, scratch2;
    __asm__ __volatile__("fxor.pi    %[op0], %[op0], %[op0] \n"
                         "flw.ps     %[op1], %[scatterValues]\n"
                         "mov.m.x    m1, zero, 0xAA \n"
                         "maskand    m1, m1, m0\n"
                         "maskxor    m0, m1, m0\n"
                         "fgw.ps     %[op0], %[op1](%[srcAddr]) \n"
                         "maskxor m0, m1, m0\n"
                         "fcvt.pw.ps %[op0], %[op0], rtz \n"
                         "fsrai.pi   %[op2], %[op0], 0x1f \n"
                         "fswizz.ps  %[op2], %[op2], 0xb1 \n"
                         "for.pi     %[op0], %[op0], %[op2] \n"
                         "fsw.ps     %[op0], %[dstAddr] \n"
                         : [ dstAddr ] "=m"( *(char (*)[32]) dstAddr),
                           [ op0] "=&f" (scratch0),
                           [ op1] "=&f" (scratch1),
                           [ op2] "=&f" (scratch2)

                         : [ srcAddr ] "r"( srcAddr), 
                           [ scatterValues ] "m" ( *(const int32_t (*)[8]) gatherValues1) ,
                           [ srcAddrMem ] "m" ( *(const char (*)[16]) srcAddr) // char[16] because max offset is 12+4
                         );
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, float>::value &&
                                        std::is_same<D, float16>::value,
                                    std::size_t>::type = 0>
  float convert(float s) {

    return s;
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, float>::value &&
                                        std::is_same<D, float16>::value,
                                    std::size_t>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *, int32_t *scatterValues) {
    float op0, op1;
    __asm__ __volatile__("flw.ps %[op1], %[scatterValues]\n"
                         "flw.ps %[op0], %[srcAddr] \n"
                         "fcvt.f16.ps %[op0], %[op0] \n"
                         "fsch.ps %[op0], %[op1](%[dstAddr]) \n"
                         : [op0] "=&f" (op0),
                           [op1] "=&f" (op1)
                         : [ dstAddr ] "r"(dstAddr),
                           [ srcAddr ] "m"  ( *(const char (*)[32]) srcAddr),
                           [ scatterValues ] "m"( *(const int32_t (*)[8])scatterValues)
                         :
                           "memory");  //TODO: replace memory clobber with output memory operand if scatter max offset is known
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, float>::value &&
                                        std::is_same<D, float>::value,
                                    std::size_t>::type = 0>
  DST convert(SRC s) {

    return (DST)s;
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, float>::value &&
                                        std::is_same<D, float>::value,
                                    std::size_t>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    float scratch;
    __asm__ __volatile__("flw.ps %[scratch], %[srcAddr]\n"
                         "fsw.ps %[scratch], %[dstAddr]\n"
                         : [dstAddr] "=m"  ( *(char (*)[32]) dstAddr),
                           [scratch] "=&f" (scratch)
                         : [srcAddr] "m"  ( *(const char (*)[32]) srcAddr)
                         );
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, float16>::value &&
                                        std::is_same<D, float>::value,
                                    std::size_t>::type = 0>
  float convert(float s) {

    return (float)s;
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, float16>::value &&
                                        std::is_same<D, float>::value,
                                    std::size_t>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    float op0, op1;
    __asm__ __volatile__("flw.ps %[op0], %[gatherValues] \n"
                         "fgh.ps %[op1], %[op0](%[srcAddr]) \n"
                         "fcvt.ps.f16 %[op1], %[op1] \n"
                         "fsw.ps %[op1], %[dstAddr] \n"
                         : [dstAddr] "=m"  ( * (char (*)[32]) dstAddr),
                           [op0] "=&f" (op0),
                           [op1] "=&f" (op1)
                         : [ srcAddr ] "r"(srcAddr),
                           [ gatherValues ] "m" (* (const int32_t (*)[8]) gatherValues)
                         : "memory"); //TODO: replace memory clobber with input memory operand if gather max offset is known
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, float16>::value &&
                                        std::is_same<D, float16>::value,
                                    std::size_t>::type = 0>
  DST convert(SRC s) {

    return (DST)s;
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, float16>::value &&
                                        std::is_same<D, float16>::value,
                                    std::size_t>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *) {
    float op0, op1;
    __asm__ __volatile__("flw.ps %[op0], %[gatherValues]\n"
                         "fgh.ps  %[op1], %[op0](%[srcAddr]) \n"
                         "fsch.ps  %[op1], %[op0](%[dstAddr]) \n"
                         : [op0] "=&f" (op0), [op1] "=&f" (op1)
                         : [ srcAddr ] "r"(srcAddr), [ dstAddr ] "r"(dstAddr),
                           [ gatherValues ] "m" ( *(const int32_t (*)[8]) gatherValues)
                         : "memory"); //TODO: replace memory clobber with input and output memory operands if gather/scatter max offset is known
  }

  // TODO Do proper conversion (currently we convert fp16 and int64 through
  // int32)
  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, int64_t>::value &&
                                        std::is_same<D, float>::value,
                                    std::size_t>::type = 0>
  DST convert(SRC s) {
    int32_t tmp = (int32_t)s;
    return (DST)tmp;
  }


  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, int64_t>::value &&
                                        std::is_same<D, float>::value,
                                    std::size_t>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues,  int32_t *scatterValues) {
    float op0, op1;
    __asm__ __volatile__("flw.ps %[op0], %[gatherValues]\n"
                         "fgw.ps %[op1], %[op0](%[srcAddr]) \n"
                         "fcvt.ps.pw %[op1], %[op1] \n"
                         "fsw.ps %[op1], %[dstAddr] \n"
                         : [op0] "=&f" (op0), [op1] "=&f" (op1),
                           [ dstAddr ] "=m"( *(char(*)[32]) dstAddr)
                         : [ srcAddr ] "r"(srcAddr),
                           [ gatherValues ] "m" ( *(const int32_t (*)[8]) gatherValues)
                         :
                           "memory");  //TODO: replace memory clobber with input or/and output memory operands if gather/scatter max offset is known
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, int64_t>::value &&
                                        std::is_same<D, int64_t>::value,
                                    std::size_t>::type = 0>
  DST convert(SRC s) {

    return (DST)s;
  }

  template <typename S = SRC, typename D = DST,
            typename std::enable_if<std::is_same<S, int64_t>::value &&
                                        std::is_same<D, int64_t>::value,
                                    std::size_t>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr,  int32_t *, int32_t *) {
    int32_t gatherValues1[] = {0, 8, 16, 24, 32, 40, 48, 56};
    int32_t gatherValues2 []= {4, 12, 20, 28, 36, 44, 52, 60};
    float gv1,gv2;
    float f0,f1;
    __asm__ __volatile__(
                         "flw.ps %[gv1], %[gatherValues1]\n"
                         "flw.ps %[gv2], %[gatherValues2]\n"
                         "fgw.ps  %[f0], %[gv2](%[srcAddr]) \n"
                         "fgw.ps  %[f1], %[gv1](%[srcAddr]) \n"
                         "fscw.ps  %[f0], %[gv2](%[dstAddr]) \n"
                         "fscw.ps  %[f1], %[gv1](%[dstAddr]) \n"
                         : [gv1] "=&f" (gv1), [gv2] "=&f" (gv2),
                           [f0] "=&f" (f0), [f1] "=&f" (f1),
                           [ dstMem ] "=m" ( *( char(*)[64]) dstAddr)
                         : [ srcAddr ] "r"(srcAddr), [ dstAddr ] "r"(dstAddr),
                           [ gatherValues1 ] "m"(*(const int32_t(*)[8]) gatherValues1),
                           [ gatherValues2 ] "m"(*(const int32_t(*)[8]) gatherValues2),
                           [ srcMem ] "m" ( *(const char(*)[64]) srcAddr)
                           );
  }
};

#endif /* CONVERTER_H */
