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

#ifndef _CONVERT_TO_INST_H_
#define _CONVERT_TO_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "utils.h" // From include/internal path
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

template <size_t bytesPerElement>
inline void copy(float source, float sourceHigh, float& destination, float& destinationHigh) {
  __asm__("for.pi %[destination], %[source], %[source]\n" : [ destination ] "=f"(destination) : [ source ] "f"(source));
  if constexpr (bytesPerElement > 4) {
    __asm__("for.pi %[destinationHigh], %[sourceHigh], %[sourceHigh]\n"
            : [ destinationHigh ] "=f"(destinationHigh)
            : [ sourceHigh ] "f"(sourceHigh));
  }
}

inline void setupDequantize(float& scale, float& offset, float scaleScalar, int32_t offsetScalar) {
  __asm__("fbcx.ps %[offset], %[offsetScalar]\n"
          "fbcx.ps %[scale], %[scaleScalar]\n"
          : [ offset ] "=&f"(offset), [ scale ] "=&f"(scale)
          : [ scaleScalar ] "r"(bitwise_copy<uint32_t>(scaleScalar)), [ offsetScalar ] "r"(offsetScalar));
}

inline void doDequantize(float& destination, float source, float scale, float offset) {
  __asm__("fsub.pi %[destination], %[source], %[offset]\n"
          "fcvt.ps.pw %[destination], %[destination]\n"
          "fmul.ps %[destination], %[destination], %[scale]\n"
          : [ destination ] "+&f"(destination)
          : [ source ] "f"(source), [ offset ] "f"(offset), [ scale ] "f"(scale));
}

inline void setupQuantize(float& scaleReciprocal, float& offset, float scaleScalar, int32_t offsetScalar) {
  __asm__("fbcx.ps %[scaleReciprocal], %[scaleScalar]\n"
          "frcp.ps %[scaleReciprocal], %[scaleReciprocal]\n"
          "fbcx.ps %[offset], %[offsetScalar]\n"
          "fcvt.ps.pw %[offset], %[offset]\n"
          : [ offset ] "=&f"(offset), [ scaleReciprocal ] "=&f"(scaleReciprocal)
          : [ scaleScalar ] "r"(bitwise_copy<uint32_t>(scaleScalar)), [ offsetScalar ] "r"(offsetScalar));
}

inline void doQuantize(float& destination, float source, float scaleReciprocal, int32_t offset) {
  __asm__("fmadd.ps %[destination], %[source], %[scaleReciprocal], %[offset]\n"
          : [ destination ] "+&f"(destination)
          : [ source ] "f"(source), [ offset ] "f"(offset), [ scaleReciprocal ] "f"(scaleReciprocal));
}

template <ElemKind srcElK, ElemKind dstElK>
inline void convert(float source, float sourceHigh, float& destination, float& destinationHigh, float srcScale,
                    float srcOffset, float dstScaleReciprocal, float dstOffset) {

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

  if constexpr (srcElK == dstElK) {
    constexpr size_t bytesPerElement = Type::getElementSize(srcElK);
    copy<bytesPerElement>(source, sourceHigh, destination, destinationHigh);
  } else if constexpr (srcElK == FloatTy and dstElK == Float16Ty) {
    __asm__("fcvt.f16.ps %[destination], %[source]\n" : [ destination ] "=f"(destination) : [ source ] "f"(source));
  } else if constexpr (srcElK == FloatTy and dstElK == BFloat16Ty) {
    __asm__("fsrli.pi %[op0], %[op0], %[bits]\n"
            : [ destination ] "=f"(destination)
            : [ source ] "f"(source), [ bits ] "i"(16));
  } else if constexpr (srcElK == FloatTy and dstElK == Int8QTy) {
    // TODO: from FloatTy to Int8QTy probably not required
  } else if constexpr (srcElK == FloatTy and dstElK == UInt8QTy) {
    // TODO: from FloatTy to UInt8QTy probably not required
  } else if constexpr (srcElK == FloatTy and dstElK == Int16QTy) {
    // TODO: from FloatTy to Int16QTy probably not required
  } else if constexpr (srcElK == FloatTy and dstElK == Int32QTy) {
    // TODO: from FloatTy to Int32QTy probably not required
  } else if constexpr (srcElK == FloatTy and dstElK == Int32ITy) {
    __asm__("fbci.ps %[destination], 0x3f000\n"
            "fsgnjn.ps %[destination], %[destination], %[source]\n"
            "fadd.ps %[destination], %[source], %[destination]\n"
            "fcvt.pw.ps %[destination], %[destination], rmm\n"
            : [ destination ] "=&f"(destination)
            : [ source ] "f"(source));
  } else if constexpr (srcElK == FloatTy and dstElK == Int64ITy) {
    convert<FloatTy, Int32ITy>(source, sourceHigh, destination, destinationHigh, srcScale, srcOffset,
                               dstScaleReciprocal, dstOffset);
    convert<Int32ITy, Int64ITy>(destination, destinationHigh, destination, destinationHigh, srcScale, srcOffset,
                                dstScaleReciprocal, dstOffset);
  } else if constexpr (srcElK == FloatTy and dstElK == UInt8FusedQTy) {
    // TODO: from FloatTy to UInt8FusedQTy probably not required
  } else if constexpr (srcElK == FloatTy and dstElK == UInt8FusedFP16QTy) {
    // TODO: from FloatTy to UInt8FusedFP16QTy probably not required
  } else if constexpr (srcElK == FloatTy and dstElK == UInt4FusedFP16QTy) {
    // TODO: from FloatTy to UInt4FusedFP16QTy probably not required
  } else if constexpr (srcElK == FloatTy and dstElK == UInt4FusedQTy) {
    // TODO: from FloatTy to UInt4FusedQTy probably not required
  } else if constexpr (srcElK == FloatTy and dstElK == BoolTy) {
    float mask;
    __asm__("fclass.ps %[mask], %[source]\n"
            "fandi.pi %[mask], %[mask], 0x18\n"
            "fbci.pi %[destination], 1\n"
            "fltu.pi %[destination], %[mask], %[destination]\n"
            "fsrli.pi %[destination], %[destination], 31\n"
            : [ mask ] "=f"(mask), [ destination ] "=f"(destination)
            : [ source ] "f"(source));
  } else if constexpr (srcElK == Float16Ty and dstElK == FloatTy) {
    __asm__("fcvt.ps.f16 %[destination], %[source]\n" : [ destination ] "=f"(destination) : [ source ] "f"(source));
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
    __asm__("fslli.pi %[destination], %[source], %[bits]\n"
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
    doDequantize(destination, source, srcScale, srcOffset);
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
    doDequantize(destination, source, srcScale, srcOffset);
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
    doDequantize(destination, source, srcScale, srcOffset);
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
    doDequantize(destination, source, srcScale, srcOffset);
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
    __asm__("fcvt.ps.pw %[destination], %[source]\n" : [ destination ] "=f"(destination) : [ source ] "f"(source));
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
    __asm__("for.pi %[destination], %[source], %[source]\n"
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
    __asm__("fbci.pi %[destination], 0\n"
            "fltu.pi %[destination], %[destination], %[source]\n"
            "fsrli.pi %[destination], %[destination], 31\n"
            : [ destination ] "=&f"(destination)
            : [ source ] "f"(source));
  } else if constexpr (srcElK == Int64ITy and dstElK == FloatTy) {
    float mask, abs, absHigh, term, weight;
    __asm__(
      // Complement source and sourceHigh when negative
      "fsrai.pi %[mask], %[sourceHigh], 31\n"
      "fxor.pi %[abs], %[mask], %[source]\n"
      "fxor.pi %[absHigh], %[mask], %[sourceHigh]\n"
      // Convert
      "fslli.pi %[term], %[abs], 16\n"
      "fsrli.pi %[term], %[term], 16\n"
      "fcvt.ps.pw %[destination], %[term]\n"
      "fbci.ps %[weight], 0x47800\n"
      "fsrli.pi %[term], %[abs], 16\n"
      "fmadd.ps %[destination], %[weight], %[term], %[destination]\n"
      "fbci.ps %[weight], 0x4f800\n"
      "fcvt.ps.pw %[term], %[absHigh]\n"
      "fmadd.ps %[destination], %[weight], %[term], %[destination]\n"
      // Add the increment
      "fsrli.pi %[mask], %[sourceHigh], 31\n"
      "fcvt.ps.pw %[mask], %[mask]\n"
      "fadd.ps %[destination], %[destination], %[mask]\n"
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
    __asm__("for.pi %[combinedOr], %[source], %[sourceHigh]\n"
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

template <ElemKind srcElK, ElemKind dstElK, bool alignedSrc, bool alignedDst>
inline void loadConvertStore(uintptr_t srcAddr, uintptr_t dstAddr, uint64_t conf, float indices, float indicesHigh,
                             uint64_t dstConf, float dstIndices, float dstIndicesHigh, float srcScale, float srcOffset,
                             float dstScaleReciprocal, float dstOffset) {

  constexpr size_t srcBytesPerElement = Type::getElementSize(srcElK);
  constexpr size_t dstBytesPerElement = Type::getElementSize(dstElK);
  constexpr bool sameConfig = isSameConfig<srcElK, dstElK, alignedSrc, alignedDst>();

  float op0 = 0.f, op0High = 0.f;
  load<srcBytesPerElement, alignedSrc>(srcAddr, conf, indices, indicesHigh, op0, op0High);

  float op1 = 0.f, op1High = 0.f;
  convert<srcElK, dstElK>(op0, op0High, op1, op1High, srcScale, srcOffset, dstScaleReciprocal, dstOffset);

  if constexpr (sameConfig) {
    store<dstBytesPerElement, alignedDst>(dstAddr, conf, indices, indicesHigh, op1, op1High);
  } else {
    store<dstBytesPerElement, alignedDst>(dstAddr, dstConf, dstIndices, dstIndicesHigh, op1, op1High);
  }
}

template <ElemKind dstElK, ElemKind srcElK>
inline __attribute__((always_inline)) void
fwdLibConvertToInstVectorized(LibTensor* outT, LibTensor* inT, uint64_t flags, const uint32_t minionOffset = 0,
                              const uint32_t assignedMinions = 0) {

  constexpr size_t srcBytesPerElement = Type::getElementSize(srcElK);
  constexpr size_t dstBytesPerElement = Type::getElementSize(dstElK);
  constexpr bool alignedSrc = false;
  constexpr bool alignedDst = false;
  constexpr bool sameConfig = isSameConfig<srcElK, dstElK, alignedSrc, alignedDst>();

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;

  if (minionId >= activeMinions) {
    return;
  }

  void* srcT = inT->getRawDataPointer<void>();
  void* dstT = outT->getRawDataPointer<void>();

  const dim_t* srcIndex = inT->dims().data();
  const dim_t* srcPitch = inT->strides().data();
  const dim_t* dstPitch = outT->strides().data();

  size_t srcDimNum = inT->ndims();

  // Total number of elements in the tensor
  unsigned int numElemsDst = dstPitch[0] * srcIndex[0];

  // Each minion does a region of maxRead consecutive elements starting at
  // initialAddr
  unsigned int initialAddr, maxRead;
  getCachelinePartition(dstBytesPerElement, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dstT);

  if (maxRead == 0) {
    return;
  }

  // Destination tensor coordinates
  unsigned int coord[srcDimNum];

  // Number of non-zero coordinates
  unsigned int k = 0;

  // We move the initialAddr to the next non-padding position
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, srcIndex, k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += srcPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n");

  uint64_t conf;
  float indices;
  float indicesHigh;
  setupGatherScatterConfig<srcBytesPerElement, alignedDst>(conf, indices, indicesHigh);

  uint64_t dstConf;
  float dstIndices;
  float dstIndicesHigh;
  if constexpr (not sameConfig) {
    setupGatherScatterConfig<dstBytesPerElement, alignedSrc>(dstConf, dstIndices, dstIndicesHigh);
  }

  float srcScale, srcOffset;
  float srcScaleScalar = inT->getScale();
  int32_t srcOffsetScalar = outT->getOffset();
  (void)srcScale;
  (void)srcOffset;
  (void)srcScaleScalar;
  (void)srcOffsetScalar;
  if constexpr (isQuantizedElemKind(srcElK)) {
    setupDequantize(srcScale, srcOffset, srcScaleScalar, srcOffsetScalar);
  }

  float dstScaleReciprocal, dstOffset;
  float dstScaleScalar = outT->getScale();
  int32_t dstOffsetScalar = outT->getOffset();
  (void)dstScaleReciprocal;
  (void)dstOffset;
  (void)dstScaleScalar;
  (void)dstOffsetScalar;
  if constexpr (isQuantizedElemKind(srcElK)) {
    setupQuantize(dstScaleReciprocal, dstOffset, dstScaleScalar, dstOffsetScalar);
  }

  unsigned int posMax = maxRead + initialAddr;

  bool done = false;
  while (not done and offsetOut < posMax) {
    uintptr_t srcAddr = (uintptr_t)srcT + offsetIn * srcBytesPerElement;
    uintptr_t dstAddr = (uintptr_t)dstT + offsetOut * dstBytesPerElement;
    loadConvertStore<srcElK, dstElK, alignedSrc, alignedDst>(srcAddr, dstAddr, conf, indices, indicesHigh, dstConf,
                                                             dstIndices, dstIndicesHigh, srcScale, srcOffset,
                                                             dstScaleReciprocal, dstOffset);
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, srcIndex, srcPitch, dstPitch);
  }

  if (DO_EVICTS) {
    unsigned int clperminion = maxRead * dstBytesPerElement / CACHE_LINE_BYTES;
    if (clperminion > 0) {
      evict_va_multi(DO_EVICTS, (uintptr_t)dstT + dstBytesPerElement * initialAddr, clperminion);
    }
  }
}

template <ElemKind dstElK, ElemKind srcElK>
inline __attribute__((always_inline)) void fwdLibConvertToInst(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                                               const uint32_t minionOffset = 0,
                                                               const uint32_t assignedMinions = 0) {
  dnn_lib::inlining::fwdLibConvertToInstVectorized<dstElK, srcElK>(outT, inT, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _CONVERT_TO_INST_H_
