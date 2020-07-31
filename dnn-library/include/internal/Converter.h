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


#include <math.h>

#include <limits>


namespace dnn_lib {

#define ONLY_FOR(cond) template< ElemKind D = dstElK, ElemKind S = srcElK,      \
                               typename std::enable_if<(cond), size_t>::type = 0>

template <ElemKind srcElK, ElemKind dstElK> class Converter {
public:
  using dstType = typename elemKind2elemTy<dstElK>::type;
  using srcType = typename elemKind2elemTy<srcElK>::type;
  Converter(){};
  

  ONLY_FOR( S == Float16Ty)
  static dstType convert(srcType val)
  {
    if (D == Float16Ty) return val;
    float fval = 0;
    convertFp16ToFp32(val, fval);
    return Converter<FloatTy, D>::convert(fval);
  }

  ONLY_FOR( S == FloatTy)
  static dstType convert(srcType val)
  {
    switch (D) {
    case ElemKind::FloatTy: {
      return val;
    } break;
    case ElemKind::Float16Ty: {
      uint16_t uint16val = 0;
      convertFp32ToFp16(val, uint16val);
      return static_cast<dstType>(uint16val);      
    } break;
    case ElemKind::Int32ITy: {
      return static_cast<dstType>(val);
    } break;
    case ElemKind::Int64ITy: {
      // @TODO Do proper conversion (currently we convert fp16 and int64 through
      // int32)
      int32_t tmpval = static_cast<int32_t>(val);
      return static_cast<dstType>(tmpval);
    } break;
    case ElemKind::BoolTy: {
      if (isnan(val)) return false; // it is not-a-number 
      return std::abs(val) > std::numeric_limits<float>::epsilon();
    } break;
    default: {
      assert(true && "Not supported conversion");
    } break;

    }
  }

  ONLY_FOR( S == Int32ITy)
  static dstType convert(srcType val)
  {
    switch (D) {
    case ElemKind::FloatTy: {
      return static_cast<dstType>(val);
    } break;
    case ElemKind::Float16Ty: {
      uint16_t uint16val = 0;
      float fval = static_cast<float>(val);
      convertFp32ToFp16(fval, uint16val);
      return static_cast<dstType>(uint16val);      
    } break;
    case ElemKind::Int32ITy: {
      return val;
    } break;
    case ElemKind::Int64ITy: {
      // @TODO Do proper conversion (currently we convert fp16 and int64 through
      // int32)
      int32_t tmpval = static_cast<int32_t>(val);
      return static_cast<dstType>(tmpval);
    } break;
    case ElemKind::BoolTy: {
      return (val != 0);
    } break;
    default: {
      assert(true && "Not supported conversion");
    } break;

    }
  }

  ONLY_FOR( S == Int64ITy)
  static dstType convert(srcType val)
  {
   switch (D) {
    case ElemKind::FloatTy: {
      int32_t tmpval = static_cast<int32_t>(val);
      return static_cast<dstType>(tmpval);
    } break;
    case ElemKind::Float16Ty: {
      uint16_t uint16val = 0;
      int32_t int32val = static_cast<int32_t>(val);
      float fval = static_cast<float>(int32val);      
      convertFp32ToFp16(fval, uint16val);
      return static_cast<dstType>(uint16val);      
    } break;
    case ElemKind::Int32ITy: {
      int32_t tmpval = static_cast<int32_t>(val);
      return static_cast<dstType>(tmpval);      
    } break;
    case ElemKind::Int64ITy: {
      return val;
    } break;
    case ElemKind::BoolTy: {
      return static_cast<dstType>(val != 0);
    } break;
    default: {
      assert(true && "Not supported conversion");
    } break;

    }
  }

  ONLY_FOR( S == BoolTy)
  static dstType convert(srcType val)
  {
   switch (D) {
    case ElemKind::FloatTy: {
      float fval =  val?1.0f:0.0f;
      return static_cast<dstType>(fval);
    } break;
    case ElemKind::Float16Ty: {
      uint16_t uint16val = 0;
      float fval = val?1.0f:0.0f;
      convertFp32ToFp16(fval, uint16val);
      return static_cast<dstType>(uint16val);      
    } break;
    case ElemKind::Int32ITy: {
      return static_cast<dstType>(val?1:0);
    } break;
    case ElemKind::Int64ITy: {
      return static_cast<dstType>(val?1:0);
    } break;
    case ElemKind::BoolTy: {
      return val;
    } break;
    default: {
      assert(true && "Not supported conversion");
    } break;

    }
  }

  template < ElemKind SRC = srcElK, ElemKind DST = dstElK,
             typename std::enable_if<SRC == FloatTy && DST == Int64ITy, int>::type = 0>
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

  
  template < ElemKind SRC = srcElK, ElemKind DST = dstElK,
             typename std::enable_if<SRC == FloatTy && DST == Float16Ty, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *, int32_t *scatterValues) {
    float op0, op1;
    __asm__ __volatile__("flw.ps %[op1], %[scatterValues]\n"
                         "flw.ps %[op0], %[srcAddr] \n"
                         "fcvt.f16.ps %[op0], %[op0] \n"
                         "fsch.ps %[op0], %[op1](%[dstAddr]) \n"
                         : [op0] "=&f" (op0),
                           [op1] "=&f" (op1),
                           "=m"  ( * (char (*)[16]) dstAddr)
                         : [ dstAddr ] "r"(dstAddr),
                           [ srcAddr ] "m"  ( *(const char (*)[32]) srcAddr),
                           [ scatterValues ] "m"( *(const int32_t (*)[8])scatterValues)
                         );
  }
  
 
  template < ElemKind SRC = srcElK, ElemKind DST = dstElK,
             typename std::enable_if<SRC == FloatTy && DST == FloatTy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    float scratch;
    __asm__ __volatile__("flw.ps %[scratch], %[srcAddr]\n"
                         "fsw.ps %[scratch], %[dstAddr]\n"
                         : [dstAddr] "=m"  ( *(char (*)[32]) dstAddr),
                           [scratch] "=&f" (scratch)
                         : [srcAddr] "m"  ( *(const char (*)[32]) srcAddr)
                         );
  }

  template < ElemKind SRC = srcElK, ElemKind DST = dstElK,
             typename std::enable_if<SRC == Float16Ty && DST == FloatTy, int>::type = 0>
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
                           [ gatherValues ] "m" (* (const int32_t (*)[8]) gatherValues),
                           "m"  ( * (char (*)[16]) srcAddr)
                         );
  }
  
 
  template < ElemKind SRC = srcElK, ElemKind DST = dstElK,
             typename std::enable_if<SRC == Float16Ty && DST == Float16Ty, int>::type = 0 >
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

  template < ElemKind SRC = srcElK, ElemKind DST = dstElK,
             typename std::enable_if<SRC == Int64ITy && DST == FloatTy, int>::type = 0>
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

  template < ElemKind SRC = srcElK, ElemKind DST = dstElK,
             typename std::enable_if<SRC == Int64ITy && DST == Int64ITy, int>::type = 0>
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
                           [ dstMem ] "=m" ( *( char(*)[CACHE_LINE_BYTES]) dstAddr)
                         : [ srcAddr ] "r"(srcAddr), [ dstAddr ] "r"(dstAddr),
                           [ gatherValues1 ] "m"(*(const int32_t(*)[8]) gatherValues1),
                           [ gatherValues2 ] "m"(*(const int32_t(*)[8]) gatherValues2),
                           [ srcMem ] "m" ( *(const char(*)[CACHE_LINE_BYTES]) srcAddr)
                           );
  }


  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == Float16Ty && DST == Int32ITy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }


  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == Int32ITy  && DST == Float16Ty, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }


  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == Float16Ty && DST == Int64ITy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }


  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == Int64ITy && DST == Float16Ty, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == Int64ITy && DST == Int32ITy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == Int32ITy && DST == Int64ITy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == FloatTy && DST == Int32ITy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == Int32ITy && DST == FloatTy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == FloatTy && DST == BoolTy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
    //FCLASS.S
    /* 0 rs1 is ?1. */
    /* 1 rs1 is a negative normal number. */
    /* 2 rs1 is a negative subnormal number. */
    /* 3 rs1 is ?0. */
    /* 4 rs1 is +0. */
    /* 5 rs1 is a positive subnormal number. */
    /* 6 rs1 is a positive normal number. */
    /* 7 rs1 is +1. */
    /* 8 rs1 is a signaling NaN. */
    /* 9 rs1 is a quiet NaN. */
  }
    
  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == BoolTy && DST == FloatTy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == Float16Ty && DST == BoolTy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == BoolTy && DST == Float16Ty, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == Int64ITy && DST == BoolTy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == BoolTy  && DST == Int64ITy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == Int32ITy && DST == BoolTy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == BoolTy  && DST == Int32ITy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }

  template <ElemKind SRC = srcElK, ElemKind DST = dstElK,
            typename std::enable_if<SRC == BoolTy  && DST == BoolTy, int>::type = 0>
  void convertVect(uintptr_t srcAddr, uintptr_t dstAddr, int32_t *gatherValues, int32_t *scatterValues) {
    //@TODO
  }
   
};

#undef ONLY_FOR
} // namespace dnn_lib

#endif /* CONVERTER_H */
