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
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, volatile int32_t *gatherValues, volatile int32_t *scatterValues) {
    volatile int32_t gatherValues1[] = {0, 0, 4, 4, 8, 8, 12, 12};
    __asm__ __volatile__("fxor.pi    f0, f0, f0 \n"
                         "flw.ps     f31, 0x0(%[scatterValues])\n"
                         "mov.m.x    m1, zero, 0x55 \n"
                         "maskand m1, m0, m1 \n"
                         "maskxor m0, m0, m1 \n"
                         "maskxor m1, m0, m1 \n"
                         "maskxor m0, m0, m1 \n"
                         "fgw.ps     f0, f31(%[srcAddr]) \n"
                         "maskand    m0, m1, m1 \n"
                         "fcvt.pw.ps f0, f0, rtz \n"
                         "fsrai.pi   f2, f0, 0x1f \n"
                         "fswizz.ps  f2, f2, 0xb1 \n"
                         "for.pi     f0, f0, f2 \n"
                         "fsw.ps     f0, 0x0(%[dstAddr]) \n"
                         :
                         : [ srcAddr ] "r"(srcAddr), [ dstAddr ] "r"(dstAddr),
                           [ scatterValues ] "r"(gatherValues1), [ gatherValues ] "r"(gatherValues)
                         : "f0", "f2", "f30", "f31", "memory");

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
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, volatile int32_t *gatherValues, volatile int32_t *scatterValues) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[scatterValues])\n"
                         "flw.ps f0, 0x0(%[srcAddr]) \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps f0, f31(%[dstAddr]) \n"
                         :
                         : [ srcAddr ] "r"(srcAddr), [ dstAddr ] "r"(dstAddr),
                           [ scatterValues ] "r"(scatterValues)
                         : "f0","f31", "memory");
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
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, volatile int32_t *gatherValues, volatile int32_t *scatterValues) {
    __asm__ __volatile__("flw.ps f0, 0x0(%[srcAddr]) \n"
                         "fsw.ps f0, 0x0(%[dstAddr]) \n"
                         :
                         : [ srcAddr ] "r"(srcAddr), [ dstAddr ] "r"(dstAddr)
                         : "f0", "memory");
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
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, volatile int32_t *gatherValues, volatile int32_t *scatterValues) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues]) \n"
                         "fgh.ps f0, f31(%[srcAddr]) \n"
                         "fcvt.ps.f16 f0, f0 \n"
                         "fsw.ps f0, 0x0(%[dstAddr]) \n"
                         :
                         : [ srcAddr ] "r"(srcAddr), [ dstAddr ] "r"(dstAddr), [ gatherValues ] "r"(gatherValues)
                         : "f0","f31", "memory");
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
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, volatile int32_t *gatherValues, volatile int32_t *scatterValues) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n"
                         "fgh.ps  f0, f31(%[srcAddr]) \n"
                         "fsch.ps  f0, f31(%[dstAddr]) \n"
                         :
                         : [ srcAddr ] "r"(srcAddr), [ dstAddr ] "r"(dstAddr),
                           [ gatherValues ] "r"(gatherValues)
                         : "f0", "f31", "memory");
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
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, volatile int32_t *gatherValues, volatile int32_t *scatterValues) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n"
                         "fgw.ps f0, f31(%[srcAddr]) \n"
                         "fcvt.ps.pw f0, f0 \n"
                         "fsw.ps f0, 0x0(%[dstAddr]) \n"
                         :
                         : [ srcAddr ] "r"(srcAddr), [ dstAddr ] "r"(dstAddr),
                           [ gatherValues ] "r"(gatherValues)
                         : "f0", "f31", "memory");
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
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, volatile int32_t *gatherValues, volatile int32_t *scatterValues) {
    volatile int32_t gatherValues1[] = {0, 8, 16, 24, 32, 40, 48, 56};
    volatile int32_t gatherValues2 []= {4, 12, 20, 28, 36, 44, 52, 60};
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues1])\n"
                         "flw.ps f30, 0x0(%[gatherValues2])\n"
                         "fgw.ps  f0, f30(%[srcAddr]) \n"
                         "fgw.ps  f1, f31(%[srcAddr]) \n"
                         "fscw.ps  f0, f30(%[dstAddr]) \n"
                         "fscw.ps  f1, f31(%[dstAddr]) \n"
                         :
                         : [ srcAddr ] "r"(srcAddr), [ dstAddr ] "r"(dstAddr),
                           [ gatherValues1 ] "r"(gatherValues1), [ gatherValues2 ] "r"(gatherValues2)
                         : "f0", "f1","f30", "f31", "memory");
  }
};

#endif /* CONVERTER_H */
