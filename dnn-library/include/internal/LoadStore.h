/*-------------------------------------------------------------------------
 * Copyright (C) 2020, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _LOADSTORE_H_
#define _LOADSTORE_H_

namespace dnn_lib {

constexpr uint64_t fg32b_conf = 0x398A418820;
constexpr uint64_t fg32h_conf = 0x76543210;

constexpr uint64_t getGatherScatterConfig(size_t bytesPerElement) {
  uint64_t result = 0;
  switch (bytesPerElement) {
  case 1:
    result = fg32b_conf;
    break;
  case 2:
    result = fg32h_conf;
    break;
  default:
    result = 0;
    break;
  }
  return result;
}

template <size_t bytesPerElement, bool aligned = false>
inline void setupGatherScatterConfig(uint64_t& conf, float& indices, float& indicesHigh) {

  static const int32_t values[] = {0 * bytesPerElement, 1 * bytesPerElement, 2 * bytesPerElement, 3 * bytesPerElement,
                                   4 * bytesPerElement, 5 * bytesPerElement, 6 * bytesPerElement, 7 * bytesPerElement};
  (void)values;

  if constexpr (bytesPerElement < 4) {
    if constexpr (aligned) {
      __asm__ __volatile__("li %[conf], %[confImm]\n"
                           : [ conf ] "=r"(conf)
                           : [ confImm ] "i"(getGatherScatterConfig(bytesPerElement)));
    } else {
      __asm__ __volatile__("flw.ps %[indices], %[values]\n"
                           : [ indices ] "=f"(indices)
                           : [ values ] "m"(*(const int32_t(*)[16])values));
    }
  } else if constexpr (bytesPerElement == 8) {
    __asm__ __volatile__("flw.ps %[indices], %[values]\n"
                         "faddi.pi %[indicesHigh], %[indices], 4\n"
                         : [ indices ] "=f"(indices), [ indicesHigh ] "=f"(indicesHigh)
                         : [ values ] "m"(*(const int32_t(*)[16])values));
  }
}

template <ElemKind srcElK, const ElemKind dstElK, bool alignedSrc, bool alignedDst> constexpr bool isSameConfig() {
  constexpr size_t srcBytesPerElement = Type::getElementSize(srcElK);
  constexpr size_t dstBytesPerElement = Type::getElementSize(dstElK);
  return srcBytesPerElement == dstBytesPerElement and alignedSrc == alignedDst;
}

template <ElemKind srcElK, ElemKind dstElK, bool alignedSrc = false, bool alignedDst>
inline void setupGatherScatterConfig(uint64_t& conf, float& indices, float& indicesHigh, uint64_t& dstConf,
                                     float& dstIndices, float& dstIndicesHigh) {

  constexpr size_t srcBytesPerElement = Type::getElementSize(srcElK);
  constexpr size_t dstBytesPerElement = Type::getElementSize(dstElK);
  constexpr bool sameConfig = isSameConfig<srcElK, dstElK, alignedSrc, alignedDst>();

  setupGatherScatterConfig<srcBytesPerElement>(conf, indices, indicesHigh);

  if constexpr (not sameConfig) {
    setupGatherScatterConfig<dstBytesPerElement, alignedSrc>(dstConf, dstIndices, dstIndicesHigh);
  }
}

template <size_t bytesPerElement, bool aligned = false>
inline void load(uintptr_t src, uint64_t conf, float indices, float indicesHigh, float& op0, float& op0High) {
  if constexpr (bytesPerElement == 1) {
    if constexpr (aligned) {
      __asm__ __volatile__("fg32b.ps %[op0], %[conf](%[src])\n"
                           : [ op0 ] "=f"(op0)
                           : [ conf ] "r"(conf), [ src ] "r"(src), [ srcMem ] "m"(*(const char(*)[8])src));
    } else {
      __asm__ __volatile__("fgb.ps %[op0], %[indices](%[src])\n"
                           : [ op0 ] "=f"(op0)
                           : [ indices ] "f"(indices), [ src ] "r"(src), [ srcMem ] "m"(*(const char(*)[8])src));
    }
  } else if constexpr (bytesPerElement == 2) {
    if constexpr (aligned) {
      __asm__ __volatile__("fg32h.ps %[op0], %[conf](%[src])\n"
                           : [ op0 ] "=f"(op0)
                           : [ conf ] "r"(conf), [ src ] "r"(src), [ srcMem ] "m"(*(const char(*)[16])src));
    } else {
      __asm__ __volatile__("fgh.ps %[op0], %[indices](%[src])\n"
                           : [ op0 ] "=f"(op0)
                           : [ indices ] "f"(indices), [ src ] "r"(src), [ srcMem ] "m"(*(const char(*)[16])src));
    }
  } else if constexpr (bytesPerElement == 4) {
    __asm__ __volatile__("flw.ps %[op0], %[src]\n" : [ op0 ] "=f"(op0) : [ src ] "m"(*(const char(*)[32])src));
  } else if constexpr (bytesPerElement == 8) {
    __asm__ __volatile__("fgw.ps %[op0], %[indices](%[src])\n"
                         "fgw.ps %[op0High], %[indicesHigh](%[src])\n"
                         : [ op0 ] "=&f"(op0), [ op0High ] "=f"(op0High)
                         : [ indices ] "f"(indices), [ indicesHigh ] "f"(indicesHigh), [ src ] "r"(src),
                           [ srcMem ] "m"(*(const char(*)[64])src));
  }
}

template <size_t bytesPerElement, bool aligned = false>
inline void store(uintptr_t dst, uint64_t conf, float indices, float indicesHigh, float op0, float op0High) {
  if constexpr (bytesPerElement == 1) {
    if constexpr (aligned) {
      __asm__ __volatile__("fsc32b.ps %[op0], %[conf](%[dst])\n"
                           : [ dstMem ] "=m"(*(char(*)[8])dst)
                           : [ op0 ] "f"(op0), [ conf ] "r"(conf), [ dst ] "r"(dst));
    } else {
      __asm__ __volatile__("fscb.ps %[op0], %[indices](%[dst])\n"
                           : [ dstMem ] "=m"(*(char(*)[8])dst)
                           : [ op0 ] "f"(op0), [ indices ] "f"(indices), [ dst ] "r"(dst));
    }
  } else if constexpr (bytesPerElement == 2) {
    if constexpr (aligned) {
      __asm__ __volatile__("fsc32h.ps %[op0], %[conf](%[dst])\n"
                           : [ dstMem ] "=m"(*(char(*)[16])dst)
                           : [ op0 ] "f"(op0), [ conf ] "r"(conf), [ dst ] "r"(dst));
    } else {
      __asm__ __volatile__("fsch.ps %[op0], %[indices](%[dst])\n"
                           : [ dstMem ] "=m"(*(char(*)[16])dst)
                           : [ op0 ] "f"(op0), [ indices ] "f"(indices), [ dst ] "r"(dst));
    }
  } else if constexpr (bytesPerElement == 4) {
    __asm__ __volatile__("fsw.ps %[op0], (%[dst])\n"
                         : [ dstMem ] "=m"(*(char(*)[32])dst)
                         : [ op0 ] "f"(op0), [ dst ] "r"(dst));
  } else if constexpr (bytesPerElement == 8) {
    __asm__ __volatile__("fscw.ps %[op0], %[indices](%[dst])\n"
                         "fscw.ps %[op0High], %[indicesHigh](%[dst])\n"
                         : [ dstMem ] "=m"(*(char(*)[64])dst)
                         : [ op0 ] "f"(op0), [ op0High ] "f"(op0High), [ indices ] "f"(indices),
                           [ indicesHigh ] "f"(indicesHigh), [ dst ] "r"(dst));
  }
}

} // namespace dnn_lib

#endif // _LOADSTORE_H_
