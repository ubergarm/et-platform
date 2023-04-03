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

#include "LibTensor.h"
#include "LibTypes.h"
#include <limits>

namespace dnn_lib {

enum class RoundingMode {
  NearestTiesEven = 0,
  TowardsZero,
  Down,
  Up,
  NearestTiesMax,
  Invalid1,
  Invalid2,
  Dynamic,
  LikeStdRoundAndCast = NearestTiesEven,
  LikeNearbyIntFAndCast = NearestTiesMax,
  LikeCast = TowardsZero
};

template <RoundingMode mode = RoundingMode::Dynamic, bool careAboutNonFinite = true,
          bool careAboutSignallingNaN = false>
INLINE_ATTR void convertFloatToInt32(float source, float& destination) {

  static_assert(mode != RoundingMode::Invalid1 and mode != RoundingMode::Invalid2);

  // When converting from FloatTy to Int32ITy, two floating point unit
  // standards like RISC-V and x86 deal differently with with -Inf, +Inf,
  // NaN and sNaN values, as shown in the following table:
  //
  // |        | -Inf       | +Inf       | NaN        | sNaN       |
  // |--------|------------|------------|------------|------------|
  // | RISC-V | 0x80000000 | 0x7fffffff | 0x7fffffff | 0x7fffffff |
  // | x86    | 0x80000000 | 0x80000000 | 0x80000000 | 0x80000000 |
  //
  // The reference for the RISC-V values in our table is the public
  // spec. Regarding to the values for x86 they were found by trial and
  // error and then verified that they matched the behaviour of the
  // legacy FIST instruction as described on the "Intel 64 and IA-32
  // Architectures Software Developer's Manual":
  // https://www.intel.com/content/dam/www/public/us/en/documents/manuals/
  // 64-ia-32-architectures-software-developer-instruction-set-reference-
  // manual-325383.pdf
  //
  // If one needs to implement a C99 style cast like "(int)(floatValue)",
  // the language standard gives the freedom of producing any value we wish
  // for the four inputs in the table. That freedom comes from the fact
  // that the standard conveniently states that convertions are undefined
  // for values outside of the range than can be represented. See section
  // 6.3.1.4 from the C99 standard here:
  // http://www.open-std.org/jtc1/sc22/WG14/www/docs/n1256.pdf
  //
  // Note that as a corolary of the wording in the C99 standard we have that
  // for producing a C99 capable CPU one does not even need to clamp values
  // above 2^31-1 or below -2^31.
  //
  // However, ETSoC goes the extra leg in our sofware by patching our RISC-V
  // style results for matching the x86 convertion. We do the fix by
  // incrementing the result when bits 7, 8 or 9 from the fclass returned
  // mask are set, meaning the input is +Inf orNaN or sNaN.
  //
  // Note that this behaviour may be optionally dropped if something like a
  // "fast math" mode is provided in the future by dnnLibrary.

  float bit;

  if constexpr (careAboutNonFinite) {
    float temp, mask;
    __asm__ __volatile__("fclass.ps %[mask], %[source]\n"
                         "fsrli.pi %[bit], %[mask], 9\n"
                         : [ mask ] "=&f"(mask), [ bit ] "=f"(bit)
                         : [ source ] "f"(source));

    if constexpr (careAboutSignallingNaN) {
      __asm__ __volatile__("fsrli.pi %[temp], %[mask], 8\n"
                           "for.pi %[bit], %[temp], %[bit]\n"
                           : [ temp ] "=&f"(temp), [ bit ] "+f"(bit)
                           : [ mask ] "f"(mask));
    }

    __asm__ __volatile__("fsrli.pi %[temp], %[mask], 7\n"
                         "for.pi %[bit], %[temp], %[bit]\n"
                         "fandi.pi %[bit], %[bit], 1\n"
                         : [ temp ] "=&f"(temp), [ bit ] "+f"(bit)
                         : [ mask ] "f"(mask));
  }

  if constexpr (mode == RoundingMode::NearestTiesEven) {
    __asm__ __volatile__("fcvt.pw.ps %[destination], %[source], rne\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source));
  } else if constexpr (mode == RoundingMode::TowardsZero) {
    __asm__ __volatile__("fcvt.pw.ps %[destination], %[source], rtz\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source));
  } else if constexpr (mode == RoundingMode::Down) {
    __asm__ __volatile__("fcvt.pw.ps %[destination], %[source], rdn\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source));
  } else if constexpr (mode == RoundingMode::Up) {
    __asm__ __volatile__("fcvt.pw.ps %[destination], %[source], rup\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source));
  } else if constexpr (mode == RoundingMode::NearestTiesMax) {
    __asm__ __volatile__("fcvt.pw.ps %[destination], %[source], rmm\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source));
  } else if constexpr (mode == RoundingMode::Dynamic) {
    __asm__ __volatile__("fcvt.pw.ps %[destination], %[source], dyn\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source));
  }

  if constexpr (careAboutNonFinite) {
    __asm__ __volatile__("fadd.pi %[destination], %[destination], %[bit]\n"
                         : [ destination ] "+f"(destination)
                         : [ bit ] "f"(bit));
  }

  (void)bit;
}

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
INLINE_ATTR void setupGatherScatterConfig(uint64_t& conf, float& indices) {

  if constexpr (aligned) {
    __asm__ __volatile__("li %[conf], %[confImm]\n"
                         : [ conf ] "=r"(conf)
                         : [ confImm ] "i"(getGatherScatterConfig(bytesPerElement)));
  } else {
    static const int32_t values[] = {0 * bytesPerElement, 1 * bytesPerElement, 2 * bytesPerElement,
                                     3 * bytesPerElement, 4 * bytesPerElement, 5 * bytesPerElement,
                                     6 * bytesPerElement, 7 * bytesPerElement};

    __asm__ __volatile__("flw.ps %[indices], %[values]\n"
                         : [ indices ] "=f"(indices)
                         : [ values ] "m"(*(const int32_t(*)[16])values));
  }
}

template <size_t bytesPerElement, bool aligned = false>
INLINE_ATTR void setupGatherScatterConfig(uint64_t& conf, float& indices, float& indicesHigh) {

  if constexpr (bytesPerElement == 8) {
    static const int32_t values[] = {0 * bytesPerElement, 1 * bytesPerElement, 2 * bytesPerElement,
                                     3 * bytesPerElement, 4 * bytesPerElement, 5 * bytesPerElement,
                                     6 * bytesPerElement, 7 * bytesPerElement};
    __asm__ __volatile__("flw.ps %[indices], %[values]\n"
                         "faddi.pi %[indicesHigh], %[indices], 4\n"
                         : [ indices ] "=f"(indices), [ indicesHigh ] "=f"(indicesHigh)
                         : [ values ] "m"(*(const int32_t(*)[16])values));
  } else if constexpr (bytesPerElement < 4) {
    setupGatherScatterConfig<bytesPerElement, aligned>(conf, indices);
  }
}

template <size_t srcBytesPerElement, size_t dstBytesPerElement, bool alignedSrc, bool alignedDst>
constexpr bool isSameConfig() {
  return srcBytesPerElement == dstBytesPerElement and alignedSrc == alignedDst;
}

template <size_t srcBytesPerElement, size_t dstBytesPerElement, bool alignedSrc = false, bool alignedDst = false>
INLINE_ATTR void setupGatherScatterConfig(uint64_t& conf, float& indices, float& indicesHigh, uint64_t& dstConf,
                                          float& dstIndices, float& dstIndicesHigh) {
  setupGatherScatterConfig<srcBytesPerElement, alignedSrc>(conf, indices, indicesHigh);
  if constexpr (not isSameConfig<srcBytesPerElement, dstBytesPerElement, alignedSrc, alignedDst>()) {
    setupGatherScatterConfig<dstBytesPerElement, alignedDst>(dstConf, dstIndices, dstIndicesHigh);
  }
}

template <size_t bytesPerElement, bool aligned = false>
INLINE_ATTR void load(uintptr_t src, [[maybe_unused]] uint64_t conf, const float& indices, float& op0) {
  static_assert(bytesPerElement == 1 or bytesPerElement == 2 or bytesPerElement == 4, "Unsupported element size");
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
  }
}

template <size_t bytesPerElement, bool aligned = false>
INLINE_ATTR void load(uintptr_t src, [[maybe_unused]] uint64_t conf, float indices, [[maybe_unused]] float indicesHigh,
                      float& op0, float& op0High) {
  if constexpr (bytesPerElement == 8) {
    __asm__ __volatile__("fgw.ps %[op0], %[indices](%[src])\n"
                         "fgw.ps %[op0High], %[indicesHigh](%[src])\n"
                         : [ op0 ] "=&f"(op0), [ op0High ] "=f"(op0High)
                         : [ indices ] "f"(indices), [ indicesHigh ] "f"(indicesHigh), [ src ] "r"(src),
                           [ srcMem ] "m"(*(const char(*)[64])src));
  } else {
    load<bytesPerElement, aligned>(src, conf, indices, op0);
  }
}

template <size_t bytesPerElement, bool aligned = false>
INLINE_ATTR void storeLocal(uintptr_t dst, [[maybe_unused]] uint64_t conf, [[maybe_unused]] float indices, float op0) {
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
    __asm__ __volatile__("fsw.ps %[op0], 0(%[dst])\n"
                         : [ dstMem ] "=m"(*(char(*)[32])dst)
                         : [ op0 ] "f"(op0), [ dst ] "r"(dst));
  }
}

template <size_t bytesPerElement, bool aligned = false>
INLINE_ATTR void storeGlobal(uintptr_t dst, [[maybe_unused]] uint64_t conf, float indices, float op0) {
  // TODO [SW-11008] aligned global stores are not optimized on this implementation.
  if constexpr (bytesPerElement == 1) {
    __asm__ __volatile__("fscbg.ps %[op0], %[indices](%[dst])\n"
                         : [ dstMem ] "=m"(*(char(*)[8])dst)
                         : [ op0 ] "f"(op0), [ indices ] "f"(indices), [ dst ] "r"(dst));
  } else if constexpr (bytesPerElement == 2) {
    __asm__ __volatile__("fschg.ps %[op0], %[indices](%[dst])\n"
                         : [ dstMem ] "=m"(*(char(*)[16])dst)
                         : [ op0 ] "f"(op0), [ indices ] "f"(indices), [ dst ] "r"(dst));
  } else if constexpr (bytesPerElement == 4) {
    __asm__ __volatile__("fswg.ps %[op0], (%[dst])\n"
                         : [ dstMem ] "=m"(*(char(*)[32])dst)
                         : [ op0 ] "f"(op0), [ dst ] "r"(dst));
  }
}

template <size_t bytesPerElement, bool aligned = false, bool globalStore = false>
INLINE_ATTR void store(uintptr_t dst, uint64_t conf, float indices, float op0) {
  static_assert(bytesPerElement == 1 or bytesPerElement == 2 or bytesPerElement == 4, "Unsupported element size");
  if constexpr (globalStore) {
    storeGlobal<bytesPerElement, aligned>(dst, conf, indices, op0);
  } else {
    storeLocal<bytesPerElement, aligned>(dst, conf, indices, op0);
  }
}

template <size_t bytesPerElement, bool aligned = false, bool globalStore = false>
INLINE_ATTR void store(uintptr_t dst, [[maybe_unused]] uint64_t conf, float indices, [[maybe_unused]] float indicesHigh,
                       float op0, [[maybe_unused]] float op0High) {
  if constexpr (bytesPerElement == 8) {
    if constexpr (globalStore) {
      // TODO [SW-11008] aligned global stores are not optimized on this implementation.
      __asm__ __volatile__("fscwg.ps %[op0], %[indices](%[dst])\n"
                           "fscwg.ps %[op0High], %[indicesHigh](%[dst])\n"
                           : [ dstMem ] "=m"(*(char(*)[64])dst)
                           : [ op0 ] "f"(op0), [ op0High ] "f"(op0High), [ indices ] "f"(indices),
                             [ indicesHigh ] "f"(indicesHigh), [ dst ] "r"(dst));
    } else {
      __asm__ __volatile__("fscw.ps %[op0], %[indices](%[dst])\n"
                           "fscw.ps %[op0High], %[indicesHigh](%[dst])\n"
                           : [ dstMem ] "=m"(*(char(*)[64])dst)
                           : [ op0 ] "f"(op0), [ op0High ] "f"(op0High), [ indices ] "f"(indices),
                             [ indicesHigh ] "f"(indicesHigh), [ dst ] "r"(dst));
    }
  } else {
    store<bytesPerElement, aligned, globalStore>(dst, conf, indices, op0);
  }
}

template <size_t bytesPerElement> INLINE_ATTR void copy(float source, float& destination) {
  __asm__ __volatile__("for.pi %[destination], %[source], %[source]\n"
                       : [ destination ] "=f"(destination)
                       : [ source ] "f"(source));
}

template <size_t bytesPerElement>
INLINE_ATTR void copy(float source, [[maybe_unused]] float sourceHigh, float& destination, float& destinationHigh) {
  copy<bytesPerElement>(source, destination);
  if constexpr (bytesPerElement > 4) {
    copy<bytesPerElement>(sourceHigh, destinationHigh);
  }
}

template <ElemKind elK> INLINE_ATTR void zero(float& destination) {
  __asm__ __volatile__("fbci.pi %[destination], 0\n" : [ destination ] "=f"(destination));
}

template <ElemKind elK> INLINE_ATTR void zero(float& destination, float& destinationHigh) {
  zero<elK>(destination);
  if constexpr (Type::getElementSize(elK) > 4) {
    zero<elK>(destinationHigh);
  }
}

INLINE_ATTR void setupDequantize(float& scale, float& offset, float scaleScalar, int32_t offsetScalar) {
  __asm__ __volatile__("fbcx.ps %[offset], %[offsetScalar]\n"
                       "fbcx.ps %[scale], %[scaleScalar]\n"
                       : [ offset ] "=&f"(offset), [ scale ] "=&f"(scale)
                       : [ scaleScalar ] "r"(bitwise_copy<uint32_t>(scaleScalar)), [ offsetScalar ] "r"(offsetScalar));
}

/*
Compute the expression "(float)(source - offset) * scale", with "source" and
"offset" being int32 and scale being float.

source: packed int32 register with source
scale:  packed float register with the broadcasted scale value
offset: packed int32 register with the broadcasted offset
*/
INLINE_ATTR void doDequantizeInt32(float& destination, float source, float scale, float offset) {
  __asm__ __volatile__("fsub.pi %[destination], %[source], %[offset]\n"
                       "fcvt.ps.pw %[destination], %[destination]\n"
                       "fmul.ps %[destination], %[destination], %[scale]\n"
                       : [ destination ] "=&f"(destination)
                       : [ source ] "f"(source), [ offset ] "f"(offset), [ scale ] "f"(scale));
}

/*
Compute the expression "(float)(source - offset) * scale", with "source" and
"offset" uint32 and scale float.

source: packed uint32 register with source
scale:  packed float register with the broadcasted scale value
offset: packed uint32 register with the broadcasted offset
*/
INLINE_ATTR void doDequantizeUInt32(float& destination, float source, float scale, float offset) {
  __asm__ __volatile__("fsub.pi %[destination], %[source], %[offset]\n"
                       "fcvt.ps.pwu %[destination], %[destination]\n"
                       "fmul.ps %[destination], %[destination], %[scale]\n"
                       : [ destination ] "=&f"(destination)
                       : [ source ] "f"(source), [ offset ] "f"(offset), [ scale ] "f"(scale));
}

INLINE_ATTR void setupQuantize(float& scaleReciprocal, float& offset, float scaleScalar, int32_t offsetScalar) {
  __asm__ __volatile__("fbcx.ps %[scaleReciprocal], %[scaleScalar]\n"
                       "frcp.ps %[scaleReciprocal], %[scaleReciprocal]\n"
                       "fbcx.ps %[offset], %[offsetScalar]\n"
                       "fcvt.ps.pw %[offset], %[offset]\n"
                       : [ offset ] "=&f"(offset), [ scaleReciprocal ] "=&f"(scaleReciprocal)
                       : [ scaleScalar ] "r"(bitwise_copy<uint32_t>(scaleScalar)), [ offsetScalar ] "r"(offsetScalar));
}

INLINE_ATTR void multiplyAdd(float& destination, float source, float scale, float offset) {
  __asm__ __volatile__("fmadd.ps %[destination], %[source], %[scale], %[offset]\n"
                       : [ destination ] "=f"(destination)
                       : [ source ] "f"(source), [ offset ] "f"(offset), [ scale ] "f"(scale));
}

template <int64_t minValue, int64_t maxValue> INLINE_ATTR void clip(float& destination, float& source) {
  if constexpr (minValue == 0 and maxValue == 255) {
    __asm__("fsatu8.pi %0, %0\n" : "+f"(destination));
  } else if constexpr (minValue == -127 and maxValue == 128) {
    __asm__("fsat8.pi %0, %0\n" : "+f"(destination));
  } else {
    float tmp;
    __asm__ __volatile__("fbci.pi %[tmp], %[minValue]\n"
                         "fmax.pi %[destination], %[source], %[tmp]\n"
                         "fbci.pi %[tmp], %[maxValue]\n"
                         "fmin.pi %[destination], %[destination], %[tmp]\n"
                         : [ destination ] "=f"(destination), [ tmp ] "=&f"(tmp)
                         : [ source ] "f"(source), [ minValue ] "i"(minValue & 0xfffff),
                           [ maxValue ] "i"(maxValue & 0xfffff));
  }
}

template <typename T> INLINE_ATTR void clip(float& destination, float& source) {
  clip<std::numeric_limits<T>::min(), std::numeric_limits<T>::max()>(destination, source);
}

template <ElemKind dstElK> INLINE_ATTR void clip(float& destination, float& source) {
  using type = typename elemKind2elemTy<dstElK>::type;
  clip<type>(destination, source);
}

template <ElemKind dstElK, bool careAboutNonFinite = false, bool canAboutSignallingNaN = false,
          RoundingMode roundingMode = RoundingMode::LikeStdRoundAndCast>
INLINE_ATTR void doQuantize(float& destination, float source, float scaleReciprocal, float offset) {
  static_assert(isQuantizedElemKind(dstElK));
  multiplyAdd(destination, source, scaleReciprocal, offset);
  convertFloatToInt32<roundingMode, careAboutNonFinite, canAboutSignallingNaN>(destination, destination);
  clip<dstElK>(destination, destination);
}

template <ElemKind srcElK, ElemKind dstElK, bool matchx86 = false,
          RoundingMode roundingMode = RoundingMode::LikeStdRoundAndCast>
INLINE_ATTR void convert(float source, [[maybe_unused]] float sourceHigh, float& destination, float& destinationHigh,
                         const float& srcScale, const float& srcOffset, const float& dstScaleReciprocal,
                         const float& dstOffset) {

  /*
  # The following python code was used for generating a skeleton for this funcion

  import itertools

  formats = [
    'FloatTy', 'Float16Ty', 'BFloat16Ty', 'Int8QTy', 'UInt8QTy', 'Int16QTy',
    'Int32QTy', 'Int32ITy', 'Int64ITy', 'UInt8FusedQTy', 'UInt8FusedFP16QTy',
    'UInt4FusedFP16QTy', 'UInt4FusedQTy', 'BoolTy', ]

  print("  if constexpr (srcElK == dstElK) {")
  print("    constexpr size_t bytesPerElement = Type::getElementSize(srcElK);")
  print("    copy<bytesPerElement>(source, sourceHigh, destination, destinationHigh);")

  for index, element in enumerate(itertools.product(formats, formats)):
    if element[0] != element[1]:
      print("  }} else if constexpr (srcElK == {0} and dstElK == {1}) {{".format(element[0], element[1]))
      if element[0] == "FloatTy" or element[1] == "FloatTy":
        print("    // TODO: from {0} to {1}".format(element[0], element[1]))
      else:
        print("    DEFAULT_CONVERT")

  print("  }")

  */

#define DEFAULT_CONVERT                                                                                                \
  convert<srcElK, FloatTy>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,  \
                           dstOffset);                                                                                 \
  convert<FloatTy, dstElK>(destination, destinationHigh, destination, destinationHigh, srcScale, srcOffset,            \
                           dstScaleReciprocal, dstOffset);

  [[maybe_unused]] constexpr bool careAboutNonFinite = true;
  [[maybe_unused]] constexpr bool canAboutSignallingNaN = true;

  if constexpr (srcElK == dstElK) {
    constexpr size_t bytesPerElement = Type::getElementSize(srcElK);
    copy<bytesPerElement>(source, sourceHigh, destination, destinationHigh);
  } else if constexpr (srcElK == FloatTy and dstElK == Float16Ty) {
    __asm__ __volatile__("fcvt.f16.ps %[destination], %[source]\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source));
  } else if constexpr (srcElK == FloatTy and dstElK == BFloat16Ty) {
    __asm__ __volatile__("fsrli.pi %[destination], %[source], %[bits]\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source), [ bits ] "i"(16));
  } else if constexpr (srcElK == FloatTy and dstElK == Int8QTy) {
    doQuantize<dstElK, careAboutNonFinite, canAboutSignallingNaN, roundingMode>(destination, source, dstScaleReciprocal,
                                                                                dstOffset);
  } else if constexpr (srcElK == FloatTy and dstElK == UInt8QTy) {
    doQuantize<dstElK, careAboutNonFinite, canAboutSignallingNaN, roundingMode>(destination, source, dstScaleReciprocal,
                                                                                dstOffset);
  } else if constexpr (srcElK == FloatTy and dstElK == Int16QTy) {
    doQuantize<dstElK, careAboutNonFinite, canAboutSignallingNaN, roundingMode>(destination, source, dstScaleReciprocal,
                                                                                dstOffset);
  } else if constexpr (srcElK == FloatTy and dstElK == Int32QTy) {
    doQuantize<dstElK, careAboutNonFinite, canAboutSignallingNaN, roundingMode>(destination, source, dstScaleReciprocal,
                                                                                dstOffset);
  } else if constexpr (srcElK == FloatTy and dstElK == Int32ITy) {
    convertFloatToInt32<RoundingMode::LikeCast>(source, destination);
  } else if constexpr (srcElK == FloatTy and dstElK == Int64ITy) {
    float mask, exponent, implicit, minusExponent, tmp, mantissa;
    __asm__ __volatile__(
      // Build exoring mask for bit-wise negating when negative
      "fsrai.pi %[mask], %[source], 31\n"
      // Extract the exponent bits
      "fslli.pi %[exponent], %[source], 1\n"
      "fsrli.pi %[exponent], %[exponent], 24\n"
      // Set the implicit mantissa bit.
      //
      // Note that one does not expect the implicit mantissa bit when the
      // exponent binary is zero (-127 when converted to integer). However, for
      // the case when the exponent is -127 the result from the convert
      // operation should be zero, and for this particular case one can still
      // set the implicit mantissa bit and get the expected zero value.
      //
      // Therefore, we unconditionally set the implicit bit, irrespective of the
      // the exponent binary value.
      "fbci.ps %[implicit], 0x80000\n"
      // Subtract 127 from the exponent binary
      "faddi.pi %[exponent], %[exponent], -127\n"
      // Extract the 23 mantissa bits stored in the source operand
      "fslli.pi %[mantissa], %[source], 8\n"
      // Add the implicit bit
      "for.pi %[mantissa], %[mantissa], %[implicit]\n"
      // Populate the destination lower 32 bits
      "fbci.pi %[minusExponent], 31\n"
      "fsub.pi %[minusExponent], %[minusExponent], %[exponent]\n"
      "fsrl.pi %[destination], %[mantissa], %[minusExponent]\n"
      "faddi.pi %[minusExponent], %[exponent], -31\n"
      "fsll.pi %[tmp], %[mantissa], %[minusExponent]\n"
      "for.pi %[destination], %[destination], %[tmp]\n"
      // Populate the destination higher 32 bits
      "fbci.pi %[minusExponent], 31+32\n"
      "fsub.pi %[minusExponent], %[minusExponent], %[exponent]\n"
      "fsrl.pi %[destinationHigh], %[mantissa], %[minusExponent]\n"
      "faddi.pi %[minusExponent], %[exponent], -31-32\n"
      "fsll.pi %[tmp], %[mantissa], %[minusExponent]\n"
      "for.pi %[destinationHigh], %[destinationHigh], %[tmp]\n"
      // Apply the exoring mask
      "fxor.pi %[destination], %[mask], %[destination]\n"
      "fxor.pi %[destinationHigh], %[mask], %[destinationHigh]\n"
      // Increment
      "fsub.pi %[destination], %[destination], %[mask]\n"
      // Add carry to the higher 32 bits
      "fbci.pi %[tmp], 0\n"
      "feq.pi %[tmp], %[tmp], %[destination]\n"
      "fand.pi %[tmp], %[tmp], %[mask]\n"
      "fsub.pi %[destinationHigh], %[destinationHigh], %[tmp]\n"
      : [ mask ] "=&f"(mask), [ exponent ] "=&f"(exponent), [ implicit ] "=&f"(implicit),
        [ minusExponent ] "=f"(minusExponent), [ tmp ] "=&f"(tmp), [ mantissa ] "=f"(mantissa),
        [ destination ] "=f"(destination), [ destinationHigh ] "=f"(destinationHigh)
      : [ source ] "f"(source));
    float accumulator, bit;
    __asm__ __volatile__(
      // Override as 0x8000 0000 0000 0000 for -Inf, Inf, NaN and sNaN
      //
      // The overriding is needed for matching the x86 implementations. For
      // details see the explanation on the code converting from FloatTy to
      // Int32ITy on the convertFloatToInt32 function.
      "fclass.ps %[mask], %[source]\n"
      "fandi.pi %[accumulator], %[mask], 1\n"
      "fsrli.pi %[bit], %[mask], 7\n"
      "for.pi %[accumulator], %[accumulator], %[bit]\n"
      "fsrli.pi %[bit], %[mask], 8\n"
      "for.pi %[accumulator], %[accumulator], %[bit]\n"
      "fsrli.pi %[bit], %[mask], 9\n"
      "for.pi %[accumulator], %[accumulator], %[bit]\n"
      "fandi.pi %[accumulator], %[accumulator], 1\n"
      "fbci.pi %[bit], 0x00000\n"
      "fcmov.ps %[destination], %[accumulator], %[bit], %[destination]\n"
      "fbci.ps %[bit], 0x80000\n"
      "fcmov.ps %[destinationHigh], %[accumulator], %[bit], %[destinationHigh]\n"
      : [ mask ] "=&f"(mask), [ accumulator ] "=&f"(accumulator), [ bit ] "=&f"(bit), [ destination ] "+f"(destination),
        [ destinationHigh ] "+f"(destinationHigh)
      : [ source ] "f"(source));
  } else if constexpr (srcElK == FloatTy and dstElK == UInt8FusedQTy) {
    // TODO: from FloatTy to UInt8FusedQTy probably not required
    assert(false);
  } else if constexpr (srcElK == FloatTy and dstElK == UInt8FusedFP16QTy) {
    // TODO: from FloatTy to UInt8FusedFP16QTy probably not required
    assert(false);
  } else if constexpr (srcElK == FloatTy and dstElK == UInt4FusedFP16QTy) {
    // TODO: from FloatTy to UInt4FusedFP16QTy probably not required
    assert(false);
  } else if constexpr (srcElK == FloatTy and dstElK == UInt4FusedQTy) {
    // TODO: from FloatTy to UInt4FusedQTy probably not required
    assert(false);
  } else if constexpr (srcElK == FloatTy and dstElK == BoolTy) {
    float mask;
    __asm__ __volatile__("fclass.ps %[mask], %[source]\n"
                         "fandi.pi %[mask], %[mask], 0x18\n"
                         "fbci.pi %[destination], 1\n"
                         "fltu.pi %[destination], %[mask], %[destination]\n"
                         "fsrli.pi %[destination], %[destination], 31\n"
                         : [ mask ] "=f"(mask), [ destination ] "=f"(destination)
                         : [ source ] "f"(source));
  } else if constexpr (srcElK == Float16Ty and dstElK == FloatTy) {
    __asm__ __volatile__("fcvt.ps.f16 %[destination], %[source]\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source));
  } else if constexpr (srcElK == Float16Ty and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == Int32ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == Int64ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Float16Ty and dstElK == BoolTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == FloatTy) {
    __asm__ __volatile__("fslli.pi %[destination], %[source], %[bits]\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source), [ bits ] "i"(16));
  } else if constexpr (srcElK == BFloat16Ty and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == Int32ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == Int64ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BFloat16Ty and dstElK == BoolTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == FloatTy) {
    doDequantizeInt32(destination, source, srcScale, srcOffset);
  } else if constexpr (srcElK == Int8QTy and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == Int32ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == Int64ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int8QTy and dstElK == BoolTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == FloatTy) {
    doDequantizeUInt32(destination, source, srcScale, srcOffset);
  } else if constexpr (srcElK == UInt8QTy and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == Int32ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == Int64ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8QTy and dstElK == BoolTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == FloatTy) {
    doDequantizeInt32(destination, source, srcScale, srcOffset);
  } else if constexpr (srcElK == Int16QTy and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == Int32ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == Int64ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int16QTy and dstElK == BoolTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == FloatTy) {
    doDequantizeInt32(destination, source, srcScale, srcOffset);
  } else if constexpr (srcElK == Int32QTy and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == Int32ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == Int64ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32QTy and dstElK == BoolTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == FloatTy) {
    __asm__ __volatile__("fcvt.ps.pw %[destination], %[source]\n"
                         : [ destination ] "=f"(destination)
                         : [ source ] "f"(source));
  } else if constexpr (srcElK == Int32ITy and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == Int64ITy) {
    __asm__ __volatile__("for.pi %[destination], %[source], %[source]\n"
                         "fsrai.pi %[destinationHigh], %[source], 31\n"
                         : [ destination ] "=&f"(destination), [ destinationHigh ] "=f"(destinationHigh)
                         : [ source ] "f"(source));
  } else if constexpr (srcElK == Int32ITy and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int32ITy and dstElK == BoolTy) {
    __asm__ __volatile__("fbci.pi %[destination], 0\n"
                         "fltu.pi %[destination], %[destination], %[source]\n"
                         "fsrli.pi %[destination], %[destination], 31\n"
                         : [ destination ] "=&f"(destination)
                         : [ source ] "f"(source));
  } else if constexpr (srcElK == Int64ITy and dstElK == FloatTy) {
    float mask, abs, absHigh, term, weight;
    __asm__ __volatile__(
      // Complement source and sourceHigh when negative
      "fsrai.pi %[mask], %[sourceHigh], 31\n"
      "fxor.pi %[abs], %[mask], %[source]\n"
      "fxor.pi %[absHigh], %[mask], %[sourceHigh]\n"
      // Convert
      "fcvt.ps.pw %[destination], %[abs]\n"
      "fbci.ps %[weight], 0x4f800\n"
      "fcvt.ps.pw %[term], %[absHigh]\n"
      "fmadd.ps %[destination], %[weight], %[term], %[destination]\n"
      // Add the increment
      "fcvt.ps.pw %[mask], %[mask]\n"
      "fsub.ps %[destination], %[destination], %[mask]\n"
      // Inject the sign
      "fsgnj.ps %[destination], %[destination], %[sourceHigh]\n"
      : [ mask ] "=&f"(mask), [ abs ] "=&f"(abs), [ absHigh ] "=&f"(absHigh), [ destination ] "=&f"(destination),
        [ term ] "=&f"(term), [ weight ] "=&f"(weight)
      : [ source ] "f"(source), [ sourceHigh ] "f"(sourceHigh));
  } else if constexpr (srcElK == Int64ITy and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int64ITy and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int64ITy and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int64ITy and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int64ITy and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int64ITy and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int64ITy and dstElK == Int32ITy) {
    convert<Int32ITy, Int32ITy>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset,
                                dstScaleReciprocal, dstOffset);
  } else if constexpr (srcElK == Int64ITy and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int64ITy and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int64ITy and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int64ITy and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == Int64ITy and dstElK == BoolTy) {
    float combinedOr;
    __asm__ __volatile__("for.pi %[combinedOr], %[source], %[sourceHigh]\n"
                         "fbci.pi %[destination], 0\n"
                         "fltu.pi %[destination], %[destination], %[combinedOr]\n"
                         "fsrli.pi %[destination], %[destination], 31\n"
                         : [ destination ] "=f"(destination), [ combinedOr ] "=f"(combinedOr)
                         : [ source ] "f"(source), [ sourceHigh ] "f"(sourceHigh));
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == FloatTy) {
    // TODO: from UInt8FusedQTy to FloatTy
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == Int32ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == Int64ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedQTy and dstElK == BoolTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == FloatTy) {
    // TODO: from UInt8FusedFP16QTy to FloatTy
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == Int32ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == Int64ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt8FusedFP16QTy and dstElK == BoolTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == FloatTy) {
    // TODO: from UInt4FusedFP16QTy to FloatTy
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == Int32ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == Int64ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == UInt4FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedFP16QTy and dstElK == BoolTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == FloatTy) {
    // TODO: from UInt4FusedQTy to FloatTy
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == Float16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == BFloat16Ty) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == Int8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == UInt8QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == Int16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == Int32QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == Int32ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == Int64ITy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == UInt8FusedQTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == UInt8FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == UInt4FusedFP16QTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == UInt4FusedQTy and dstElK == BoolTy) {
    DEFAULT_CONVERT
  } else if constexpr (srcElK == BoolTy and dstElK == FloatTy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == Float16Ty) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == BFloat16Ty) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == Int8QTy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == UInt8QTy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == Int16QTy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == Int32QTy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == Int32ITy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == Int64ITy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == UInt8FusedQTy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == UInt8FusedFP16QTy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == UInt4FusedFP16QTy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  } else if constexpr (srcElK == BoolTy and dstElK == UInt4FusedQTy) {
    convert<Int32ITy, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                              dstOffset);
  }
#undef DEFAULT_CONVERT
}

template <ElemKind srcElK, ElemKind dstElK>
INLINE_ATTR void convert(float source, float sourceHigh, float& destination, float& destinationHigh) {
  static_assert(not isQuantizedElemKind(srcElK) and not isQuantizedElemKind(dstElK),
                "Quantized types are not supported by this simplified convert");
  float srcScale = 0, srcOffset = 0, dstScaleReciprocal = 0, dstOffset = 0;
  convert<srcElK, dstElK>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset, dstScaleReciprocal,
                          dstOffset);
}

template <ElemKind srcElK, ElemKind dstElK> INLINE_ATTR void convert(float& destination, float& destinationHigh) {
  if constexpr (srcElK != dstElK) {
    convert<srcElK, dstElK>(destination, destinationHigh, destination, destinationHigh);
  }
}

template <ElemKind srcElK, ElemKind dstElK> INLINE_ATTR void convert(float& destination) {
  if constexpr (srcElK != dstElK) {
    float destinationHigh = 0;
    convert<srcElK, dstElK>(destination, destinationHigh, destination, destinationHigh);
  }
}

INLINE_ATTR void saturateInt8(float source, float& destination) {
  __asm__ __volatile__("fsat8.pi %[destination], %[source]\n"
                       : [ destination ] "=f"(destination)
                       : [ source ] "f"(source));
}

INLINE_ATTR void saturateUInt8(float source, float& destination) {
  __asm__ __volatile__("fsatu8.pi %[destination], %[source]\n"
                       : [ destination ] "=f"(destination)
                       : [ source ] "f"(source));
}

} // namespace dnn_lib

#endif // _LOADSTORE_H_
