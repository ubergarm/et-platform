/*-------------------------------------------------------------------------
 * Copyright (C) 2018, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <math.h>
#include <string.h>

#include "LibNodes.h"
#include "Float16.h"
#include "LibCommon.h"
#include "device_common.h"

#define print(s) \
      ecall(FW_SCODE_ECALL_LOG_WRITE, (uint64_t)s, sizeof(s),0)

using namespace dnn_lib;
using namespace std;

#define GEN_INSTANCES(functionName, op, ...)                                   \
  template void functionName<float, op>(__VA_ARGS__);                          \
  template void functionName<float16, op>(__VA_ARGS__);                        \
  template void functionName<int8_t, op>(__VA_ARGS__);                         \
  template void functionName<int64_t, op>(__VA_ARGS__);

#define GEN_OP(functionName, ...)                                              \
  template void functionName<float>(__VA_ARGS__);                              \
  template void functionName<float16>(__VA_ARGS__);                            \
  template void functionName<int8_t>(__VA_ARGS__);                             \
  template void functionName<int64_t>(__VA_ARGS__);                            \
  template void functionName<int32_t>(__VA_ARGS__);                            \
  template void functionName<int16_t>(__VA_ARGS__);

#define GEN_3TYPE(functionName, op, ...)                                   \
  template void functionName<float, float, float, op>(__VA_ARGS__);            \
  template void functionName<float16, float16, float16, op>(__VA_ARGS__);      \
  template void functionName<int8_t, int8_t, int8_t, op>(__VA_ARGS__);         \
  template void functionName<uint8_t, int8_t, int8_t, op>(__VA_ARGS__);        \
  template void functionName<int8_t, uint8_t, int8_t, op>(__VA_ARGS__);        \
  template void functionName<int8_t, int8_t, uint8_t, op>(__VA_ARGS__);        \
  template void functionName<uint8_t, uint8_t, int8_t, op>(__VA_ARGS__);       \
  template void functionName<uint8_t, int8_t, uint8_t, op>(__VA_ARGS__);       \
  template void functionName<int8_t, uint8_t, uint8_t, op>(__VA_ARGS__);       \
  template void functionName<uint8_t, uint8_t, uint8_t, op>(__VA_ARGS__);      \
  template void functionName<int64_t, int64_t, int64_t, op>(__VA_ARGS__);      
                                                                               
#define GEN_2TYPE(functionName, op, ...)                                       \
  template void functionName<float, float, op>(__VA_ARGS__);                   \
  template void functionName<float16, float16, op>(__VA_ARGS__);               \
  template void functionName<int8_t, int8_t, op>(__VA_ARGS__);                 \
  template void functionName<uint8_t, int8_t, op>(__VA_ARGS__);                \
  template void functionName<int8_t, uint8_t, op>(__VA_ARGS__);                \
  template void functionName<uint8_t, uint8_t, op>(__VA_ARGS__);               \
  template void functionName<int64_t, int64_t, op>(__VA_ARGS__);               

#define GEN_3TYPE_OP(functionName, ...)                                        \
  template void functionName<float, float, float>(__VA_ARGS__);                \
  template void functionName<float16, float16, float16>(__VA_ARGS__);          \
  template void functionName<int8_t, int8_t, int8_t>(__VA_ARGS__);             \
  template void functionName<uint8_t, int8_t, int8_t>(__VA_ARGS__);            \
  template void functionName<int8_t, uint8_t, int8_t>(__VA_ARGS__);            \
  template void functionName<int8_t, int8_t, uint8_t>(__VA_ARGS__);            \
  template void functionName<uint8_t, uint8_t, int8_t>(__VA_ARGS__);           \
  template void functionName<uint8_t, int8_t, uint8_t>(__VA_ARGS__);           \
  template void functionName<int8_t, uint8_t, uint8_t>(__VA_ARGS__);           \
  template void functionName<uint8_t, uint8_t, uint8_t>(__VA_ARGS__);          \
  template void functionName<int64_t, int64_t, int64_t>(__VA_ARGS__);          
                                                                               
#define GEN_2TYPE_OP(functionName, ...)                                        \
  template void functionName<float, float>(__VA_ARGS__);                       \
  template void functionName<float16, float16>(__VA_ARGS__);                   \
  template void functionName<int8_t, int8_t>(__VA_ARGS__);                     \
  template void functionName<uint8_t, int8_t>(__VA_ARGS__);                    \
  template void functionName<int8_t, uint8_t>(__VA_ARGS__);                    \
  template void functionName<uint8_t, uint8_t>(__VA_ARGS__);                   \
  template void functionName<int64_t, int64_t>(__VA_ARGS__);                   

#define GEN_INTONLY_OP(functionName, ...)                                      \
  template void functionName<int64_t>(__VA_ARGS__);                            \
  template void functionName<int32_t>(__VA_ARGS__);

#define GEN_QUANT(functionName, ...)                                           \
  template void functionName<int8_t>(__VA_ARGS__);                             \
  template void functionName<int16_t>(__VA_ARGS__);                            \
  template void functionName<int32_t>(__VA_ARGS__);

#define GEN_OP_INDEX(functionName, ...)                                        \
  template void functionName<float, int64_t>(__VA_ARGS__);                     \
  template void functionName<float16, int64_t>(__VA_ARGS__);                   \
  template void functionName<int8_t, int64_t>(__VA_ARGS__);                    \
  template void functionName<int64_t, int64_t>(__VA_ARGS__);                   \
  template void functionName<int32_t, int64_t>(__VA_ARGS__);                   \
  template void functionName<float, int32_t>(__VA_ARGS__);                     \
  template void functionName<float16, int32_t>(__VA_ARGS__);                   \
  template void functionName<int8_t, int32_t>(__VA_ARGS__);                    \
  template void functionName<int64_t, int32_t>(__VA_ARGS__);                   \
  template void functionName<int32_t, int32_t>(__VA_ARGS__);

#define GEN_CONVERT(functionName, ...)                                         \
  template void functionName<float, int64_t>(__VA_ARGS__);                     \
  template void functionName<float, float16>(__VA_ARGS__);                     \
  template void functionName<float, float>(__VA_ARGS__);                       \
  template void functionName<float16, float>(__VA_ARGS__);                     \
  template void functionName<float16, float16>(__VA_ARGS__);                   \
  template void functionName<int64_t, float>(__VA_ARGS__);                     \
  template void functionName<int64_t, int64_t>(__VA_ARGS__);

namespace dnn_lib {

#include "AutoGenInstan.def"

template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized<float>(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdst2, void *pdst2Pitches, 
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, uint64_t flags,
    const uint32_t minionOffset, const uint32_t numShires);

template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized<float16>(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdst2, void *pdst2Pitches, 
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, uint64_t flags,
    const uint32_t minionOffse, const uint32_t numShires);

template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyOptimized<float>(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, uint64_t flags,
    const uint32_t minionOffset, const uint32_t numShires);

template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyOptimized<float16>(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, uint64_t flags,
    const uint32_t minionOffse, const uint32_t numShires);

}

#define M_1_LOG2E float(1.0f / M_LOG2E)
#define MAX_TENSOR_DIMENSIONS 6
#define ACTIVE_SHIRES ((flags & 0x1F) + 1)
#define DO_EVICTS                                                              \
  ((flags & 0x60) >> 5) // 01 = evictL2, 10 = evictL3, 11 = evictMem

#define SET_FG32B_VAL(_reg) "li " #_reg ", 0x398A418820\n"
#define SET_FG32H_VAL(_reg) "li " #_reg ", 0x76543210\n"
//   No SET_FG32W.PS_VAL(_reg) because flw.ps is used instead of fg32w.ps.

#define SET_MINUS_INFTY(_reg) "fbci.ps " #_reg ", 0xff800 \n" // _reg is vect
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

inline __attribute__((always_inline)) void
fpReciprocalSingleElement(float val, float &recval) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "frcp.ps %[recval], %[val] \n"
                       : [ recval ] "=&f"(recval)
                       : [ val ] "f"(val));
}

inline __attribute__((always_inline)) void
fpPowSingleElement(float val1, float val2, float &res) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "flog.ps %[res], %[val1] \n"
                       "fmul.ps %[res], %[res], %[val2] \n"
                       "fexp.ps %[res], %[res] \n"
                       : [ res ] "=&f"(res)
                       : [ val1 ] "f"(val1), [ val2 ] "f"(val2));
}

inline __attribute__((always_inline)) void fpLog2SingleElement(float val,
                                                               float &res) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "flog.ps %[res], %[val] \n"
                       : [ res ] "=&f"(res)
                       : [ val ] "f"(val));
}

inline __attribute__((always_inline)) void
fpAddSingleElement(float val1, float val2, float &res) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fadd.ps %[res], %[val1], %[val2] \n"
                       : [ res ] "=&f"(res)
                       : [ val1 ] "f"(val1), [ val2 ] "f"(val2));
}

inline __attribute__((always_inline)) void loadFp32FromMemory(uint64_t addr,
                                                              float &res) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "flw.ps %[res], 0(%[addr]) \n"
                       : [ res ] "=&f"(res)
                       : [ addr ] "r"(addr));
}

inline __attribute__((always_inline)) void convertFp16ToFp32(float src,
                                                             float &dst) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.ps.f16 %[dst], %[src] \n"
                       : [ dst ] "=&f"(dst)
                       : [ src ] "f"(src));
}

inline __attribute__((always_inline)) void convertFp16ToFp32(uint16_t src,
                                                             float &dst) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.ps.f16 %[dst], %[src] \n"
                       : [ dst ] "=&f"(dst)
                       : [ src ] "f"((float)src));
}

inline __attribute__((always_inline)) void convertFp32ToFp16(float src,
                                                             float &dst) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.f16.ps %[dst], %[src] \n"
                       : [ dst ] "=&f"(dst)
                       : [ src ] "f"(src));
}
inline __attribute__((always_inline)) void convertFp32ToFp16(float src,
                                                             uint16_t &dst) {
  float tmp;
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fcvt.f16.ps %[tmp], %[src] \n"
                       "fmv.x.w %[dst], %[tmp] \n"
                       : [ dst ] "=&r"(dst)
                       : [ src ] "f"(src), [ tmp ] "f"(tmp));
}

inline __attribute__((always_inline)) void storeFp32ToMemory(uint64_t addr,
                                                             float val32) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "fsw %[val32], 0(%[addr])\n"
                       :
                       : [ val32 ] "f"(val32), [ addr ] "r"(addr));
}

inline __attribute__((always_inline)) void storeFp16ToMemory(uint64_t addr,
                                                             float val32) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       //"fsrli.pi %[val32], %[val32], 16 \n"
                       "fmvz.x.ps x1, %[val32], 0 \n"
                       "sh x1, 0(%[addr])\n"
                       :
                       : [ val32 ] "f"(val32), [ addr ] "r"(addr));
}

float loadAndConvertToFp32(uint64_t loadAddr) {
  float val16, val32;
  loadFp32FromMemory(loadAddr, val16);
  convertFp16ToFp32(val16, val32);
  return val32;
}

void storeFp32(uint64_t storeAddr, float val32) {
  storeFp32ToMemory(storeAddr, val32);
}

float loadAndConvertToFp16(uint64_t loadAddr) {
  float val16, val32;
  loadFp32FromMemory(loadAddr, val16);
  convertFp32ToFp16(val16, val32);
  return val32;
}

void storeFp16(uint64_t storeAddr, float val32) {
  storeFp16ToMemory(storeAddr, val32);
}

inline __attribute__((always_inline)) void getReciprocal(float val,
                                                         float &recval) {
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n"
                       "frcp.ps %[recval], %[val] \n"
                       : [ recval ] "=f"(recval)
                       : [ val ] "f"(val));
}

inline __attribute__((always_inline)) unsigned int gcd(unsigned int a,
                                                       unsigned int b) {
  if (b == 1)
    return a;
  return gcd(b, a % b);
}

template<class T>
constexpr std::size_t getsize() {
  return sizeof(T);
}  
template<>
constexpr std::size_t getsize<float16>() {
  return 2;
}

inline __attribute__((always_inline)) void
getCoordinates(unsigned int *coord, unsigned int address,
                unsigned int srcDimNum, unsigned int *pitch) {
  unsigned int rm = address;
  for (unsigned int i = 0; i < srcDimNum; i++) {
    coord[i] = rm / pitch[i];
    rm = rm - coord[i] * pitch[i];
  }
}

inline __attribute__((always_inline)) void
getNonPaddingCoordinates(unsigned int *coord, unsigned int address,
                         unsigned int srcDimNum, unsigned int *pitch,
                         unsigned int *Dims, unsigned int &k) {
  getCoordinates(coord, address, srcDimNum, pitch);
  k = srcDimNum; // If it is a padding position we compute next useful position
  for (int j = srcDimNum - 1; j > 0; j--) {
    if (unlikely(coord[j] >= Dims[j])) {
      coord[j - 1]++;
      k = j;
    }
  }
  for (unsigned int j = k; j < srcDimNum; j++)
    coord[j] = 0;
}

// Power saving partition. Being used right now.
inline __attribute__((always_inline)) void
getCachelinePartition(unsigned int elementsize, unsigned int numElemsDst,
                      unsigned int &address, unsigned int &maxRead,
                      unsigned int minionId, unsigned int activeMinions) {
  unsigned int cll = 64 / elementsize;            // 64/element_size_in_bytes
  unsigned int ncl = (numElemsDst - 1) / cll + 1; // Total amount of cache lines
  unsigned int mcl = (ncl - 1) / activeMinions + 1; // Amount of cl for a minion
  unsigned int div = ncl / mcl; // div + 1 = number of active minions.
  if (minionId < div) {
    maxRead = mcl * cll;
    address = maxRead * minionId;
  } else if (minionId == div) {
    maxRead = (ncl - div * mcl) * cll;
    address = mcl * cll * minionId;
  } else
    maxRead = 0;
}

// Power wasting partition. Not being used.
inline __attribute__((always_inline)) void
getUniformCachelinePartition(unsigned int elementsize, unsigned int numElemsDst,
                             unsigned int &address, unsigned int &maxRead,
                             unsigned int activeMinions) {
  unsigned int minionId = get_minion_id();
  unsigned int cll = 64 / elementsize;            // 64/(Element Size in bytes)
  unsigned int ncl = (numElemsDst - 1) / cll + 1; // Total amount of cache lines
  unsigned int mcl =
      ncl / activeMinions; // Amount of cache lines to do for the minion
  unsigned int mod = ncl - activeMinions * mcl;
  if (minionId < mod) {
    ++mcl;
    address = mcl * cll * minionId;
  }
  else {
    address = (mod + minionId * mcl) * cll;
  }
  maxRead = mcl * cll;
}

inline __attribute__((always_inline)) void
getReversedCachelinePartition(unsigned int elementsize, unsigned int numElemsDst,
                              unsigned int &address, unsigned int &maxRead,
                              unsigned int activeMinions) {
  unsigned int minionId = (activeMinions - get_minion_id()) - 1;
  unsigned int cll = 64 / elementsize;            // 64/(Element Size in bytes)
  unsigned int ncl = (numElemsDst - 1) / cll + 1; // Total amount of cache lines
  unsigned int mcl =
      ncl / activeMinions; // Amount of cache lines to do for the minion
  unsigned int mod = ncl - activeMinions * mcl;
  if (minionId < mod) {
    ++mcl;
    address = mcl * cll * minionId;
  } else
    address = (mod + minionId * mcl) * cll;
  maxRead = mcl * cll;
}

template <typename mytype> // getOffsets may take as offsetAddr unsigned
                           // integers or uint64_t
inline __attribute__((always_inline)) bool
getOffsets(unsigned int DimNum, unsigned int *coord, mytype &offsetAddr,
           unsigned int *Index, unsigned int *Pitch) {
  for (int j = DimNum - 1; j >= 0; j--) {
    if (likely(coord[j] != (Index[j] - 1))) {
      offsetAddr += Pitch[j];
      coord[j]++;
      return false;
    } else if (likely(j != 0)) {
      offsetAddr -= (Index[j] - 1) * Pitch[j];
      coord[j] = 0;
    } else
      return true;
  }
  return true;
}

template <typename mytype> // getOffsets may take as offsetAddr unsigned
                           // integers or uint64_t
inline __attribute__((always_inline)) bool
getOffsets(unsigned int DimNum, unsigned int *coord, mytype &offsetAddr1,
           mytype &offsetAddr2, unsigned int *Index, unsigned int *pitch1,
           unsigned int *pitch2) {
  for (int j = DimNum - 1; j >= 0; j--) {
    if (likely(coord[j] != (Index[j] - 1))) {
      offsetAddr1 += pitch1[j];
      offsetAddr2 += pitch2[j];
      coord[j]++;
      return false;
    } else if (likely(j != 0)) {
      offsetAddr1 -= (Index[j] - 1) * pitch1[j];
      offsetAddr2 -= (Index[j] - 1) * pitch2[j];
      coord[j] = 0;
    } else
      return true;
  }
  return true;
}

template <typename mytype> 
                           
inline __attribute__((always_inline)) bool
getOffsets(unsigned int DimNum, unsigned int *coord, mytype &offsetAddr1,
           mytype &offsetAddr2, mytype &offsetAddr3, unsigned int *Index, unsigned int *pitch1,
           unsigned int *pitch2, unsigned int *pitch3) {
  for (int j = DimNum - 1; j >= 0; j--) {
    if (likely(coord[j] != (Index[j] - 1))) {
      offsetAddr1 += pitch1[j];
      offsetAddr2 += pitch2[j];
      offsetAddr3 += pitch3[j];
      coord[j]++;
      return false;
    } else if (likely(j != 0)) {
      offsetAddr1 -= (Index[j] - 1) * pitch1[j];
      offsetAddr2 -= (Index[j] - 1) * pitch2[j];
      offsetAddr3 -= (Index[j] - 1) * pitch3[j];
      coord[j] = 0;
    } else
      return true;
  }
  return true;
}


template <typename mytype> 
                           
inline __attribute__((always_inline)) bool
getOffsets(unsigned int DimNum, unsigned int *coord, mytype &offsetAddr1,
           mytype &offsetAddr2, mytype &offsetAddr3, mytype &offsetAddr4,
           unsigned int *Index, unsigned int *pitch1, unsigned int *pitch2,
           unsigned int *pitch3, unsigned int *pitch4) {
  for (int j = DimNum - 1; j >= 0; j--) {
    if (coord[j] != (Index[j] - 1)) {
      offsetAddr1 += pitch1[j];
      offsetAddr2 += pitch2[j];
      offsetAddr3 += pitch3[j];
      offsetAddr4 += pitch4[j];
      coord[j]++;
      return false;
    } else if (j != 0) {
      offsetAddr1 -= (Index[j] - 1) * pitch1[j];
      offsetAddr2 -= (Index[j] - 1) * pitch2[j];
      offsetAddr3 -= (Index[j] - 1) * pitch3[j];
      offsetAddr4 -= (Index[j] - 1) * pitch4[j];
      coord[j] = 0;
    } else
      return true;
  }
  return true;
}


inline __attribute__((always_inline)) bool
getOffsets(unsigned int dimNum, unsigned int *coord,
           unsigned int &offsetAddr1, unsigned int &offsetAddr2,
           unsigned int &offsetAddr3, unsigned int *index, unsigned int *pitch1,
           unsigned int *pitch2, unsigned int *pitch3) {
  for (int j = dimNum - 1; j >= 0; j--) {
    if (likely(coord[j] != (index[j] - 1))) {
      offsetAddr1 += pitch1[j];
      offsetAddr2 += pitch2[j];
      offsetAddr3 += pitch3[j];
      coord[j]++;
      return false;
    } else if (likely(j != 0)) {
      offsetAddr1 -= (index[j] - 1) * pitch1[j];
      offsetAddr2 -= (index[j] - 1) * pitch2[j];
      offsetAddr3 -= (index[j] - 1) * pitch3[j];
      coord[j] = 0;
    } else
      return true;
  }
  return true;
}

float getExp(float val) {
  float ret;
  if (val == 0) {
    return 1;
  } else if (val > 0) {
    fpPowSingleElement(M_E, val, ret);
    return ret;
  } else { // val<0
    fpPowSingleElement(M_E, -val, ret);
    fpReciprocalSingleElement(ret, ret);
    return ret;
  }
}

float getSinh(float val) {
  float op1, op2;
  if (val == 0) {
    op1 = op2 = 1;
  } else if (val > 0) {
    fpPowSingleElement(M_E, val, op1);
    fpReciprocalSingleElement(op1, op2);
  } else { // val<0
    fpPowSingleElement(M_E, -val, op2);
    fpReciprocalSingleElement(op2, op1);
  }
  return 0.5 * (op1 - op2);
}

float getCosh(float val) {
  float op1, op2;
  if (val == 0) {
    op1 = op2 = 1;
  } else if (val > 0) {
    fpPowSingleElement(M_E, val, op1);
    fpReciprocalSingleElement(op1, op2);
  } else { // val<0
    fpPowSingleElement(M_E, val, op2);
    fpReciprocalSingleElement(op2, op1);
  }
  return 0.5 * (op1 + op2);
}

bool isInteger(float f) { return (nearbyintf(f) == f); }

bool isEven(int val) { return ((val % 2) == 0); }

float getPow(float base, float exp) {
  float dst_tmp, inverted;
  if (base == 0)
    return (exp==0);
  else if ((base < 0) && !isInteger(exp))
    return NAN;
  else if ((base > 0) && exp == 0)
    return 1;
  else if (exp > 0) {
    if (base < 0) {
      fpPowSingleElement(-base, exp, dst_tmp);
      if (isEven((int)exp))
        return dst_tmp;
      else
        return -dst_tmp;
    } else {
      fpPowSingleElement(base, exp, dst_tmp);
      return dst_tmp;
    }
  } else if (exp < 0) {
    fpPowSingleElement(base, -exp, inverted);
    fpReciprocalSingleElement(inverted, dst_tmp);
    return dst_tmp;
  } else {
    return 0;
  }
}

///////////////////////////Classes for template management////////////////
// TODO
template <typename T> class Writer {
public:
  uint16_t *ptrfp16_;
  int8_t *ptri8_;
  uint8_t *ptrui8_;
  int16_t *ptri16_;
  float scale_;
  int offset_;

  template <typename U = T,
            typename std::enable_if<std::is_same<U, float16>::value,
                                    std::size_t>::type = 0>
  Writer &operator=(float value) {
    float16 f;
    f.data_ = value;
    float a = f.convertFp32ToFp16();
    *ptrfp16_ = (*(uint16_t *)&a);
    return *this;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, int8_t>::value,
                                    std::size_t>::type = 0>
  Writer &operator=(float value) {
    *ptri8_ = quantize<int8_t>(value, scale_, offset_);
    return *this;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, uint8_t>::value,
                                    std::size_t>::type = 0>
  Writer &operator=(float value) {
    *ptrui8_ = quantize<uint8_t>(value, scale_, offset_);
    return *this;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, int16_t>::value,
                                    std::size_t>::type = 0>
  Writer &operator=(float value) {
    *ptri16_ = quantize<int16_t>(value, scale_, offset_);
    return *this;
  }
};

template <typename T> class Addresser {
  T *ptrT_;
  uint16_t *ptrfp16_;
  Writer<T> writer;
  float16 utilfp16;

  float scale_;
  int32_t offset_;

public:
  Addresser(void *ptr, float scale = 1.0, int32_t offset = 0) {
    if (std::is_same<T, float16>::value == true) {
      ptrfp16_ = (uint16_t *)ptr;
    } else if (std::is_same<T, int8_t>::value == true) {
      scale_ = scale;
      offset_ = offset;
      writer.scale_ = scale;
      writer.offset_ = offset;
      ptrT_ = (T *)ptr;
    } else if (std::is_same<T, uint8_t>::value == true) {
      scale_ = scale;
      offset_ = 0;
      writer.scale_ = scale;
      writer.offset_ = 0;
      ptrT_ = (T *)ptr;
    } else if (std::is_same<T, int16_t>::value == true) {
      scale_ = scale;
      offset_ = offset;
      writer.scale_ = scale;
      writer.offset_ = offset;
      ptrT_ = (T *)ptr;
    } else {
      ptrT_ = (T *)ptr;
    }
  }

  // READ
  template <typename U = T,
            typename std::enable_if<std::is_same<U, float16>::value,
                                    std::size_t>::type = 0>
  float operator[](const int index) const {
    float16 f = (*((float *)&ptrfp16_[index]));
    return f.convertFp16ToFp32();
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, float>::value,
                                    std::size_t>::type = 0>
  const T &operator[](const int index) const {
    return ptrT_[index];
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, int8_t>::value,
                                    std::size_t>::type = 0>
  float operator[](const int index) const {
    float i32 = dequantize<int8_t>(ptrT_[index], scale_, offset_);
    return i32;
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, uint8_t>::value,
                                    std::size_t>::type = 0>
  float operator[](const int index) const {
    float i32 = dequantize<uint8_t>(ptrT_[index], scale_, offset_);
    return i32;
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, int16_t>::value,
                                    std::size_t>::type = 0>
  float operator[](const int index) const {
    float i32 = dequantize<int16_t>(ptrT_[index], scale_, offset_);
    return i32;
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, int64_t>::value,
                                    std::size_t>::type = 0>
  const T &operator[](const int index) const {
    return ptrT_[index];
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, int32_t>::value,
                                    std::size_t>::type = 0>
  const T &operator[](const int index) const {
    return ptrT_[index];
  }

  // write
  template <typename U = T,
            typename std::enable_if<std::is_same<U, float16>::value,
                                    std::size_t>::type = 0>
  Writer<T> &operator[](const int index) {
    writer.ptrfp16_ = &ptrfp16_[index];
    return writer;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, float>::value,
                                    std::size_t>::type = 0>
  T &operator[](const int index) {
    return ptrT_[index];
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, int8_t>::value,
                                    std::size_t>::type = 0>
  Writer<T> &operator[](const int index) {
    writer.ptri8_ = &ptrT_[index];
    return writer;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, uint8_t>::value,
                                    std::size_t>::type = 0>
  Writer<T> &operator[](const int index) {
    writer.ptrui8_ = &ptrT_[index];
    return writer;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, int16_t>::value,
                                    std::size_t>::type = 0>
  Writer<T> &operator[](const int index) {
    writer.ptri16_ = &ptrT_[index];
    return writer;
  }

  template <typename U = T,
            typename std::enable_if<std::is_same<U, int64_t>::value,
                                    std::size_t>::type = 0>
  T &operator[](const int index) {
    return ptrT_[index];
  }
  template <typename U = T,
            typename std::enable_if<std::is_same<U, int32_t>::value,
                                    std::size_t>::type = 0>
  T &operator[](const int index) {
    return ptrT_[index];
  }
};

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

#define OPERATION_STEP1   \
           "flw.ps f31, 0x0(%[gatherValues])\n"          \
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

template <typename src1Type, typename src2Type, typename dstType, typename opType> class Operator {
public:
  template <typename U = opType, typename S = src1Type, 
            typename enable_if<!std::is_same<S, Addresser<float16>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<int8_t>>::value && !std::is_same<S, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {


  }

  template <typename U = opType, typename S = src1Type,
            typename enable_if<!std::is_same<S, Addresser<float16>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, float *scale, int32_t *offset) {


  }

  template <typename U = opType, typename S = src1Type,
            typename enable_if<!std::is_same<S, Addresser<float16>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr, uintptr_t dstAddr, float *scale, int32_t *offset) {


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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                         "fgh.ps  f0, f31(%[src1]) \n"  
                         "fcvt.ps.f16 f0, f0 \n"  
                         "fgh.ps  f1, f31(%[src2]) \n"  
                         "fcvt.ps.f16 f1, f1 \n"  
                         "fadd.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }

  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps  f0, 0x0(%[src1]) \n" 
                         "flw.ps  f1, 0x0(%[src2]) \n"  
                         "fadd.ps f0, f0, f1 \n"
                         "fsw.ps  f0, 0x0(%[dst]) \n"
                         :
                         : [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr)
                         : "f0", "f1", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>

  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         "fmadd.ps f0, f0, f29, f1 \n" 
                         "fbc.ps f30, 0x8(%[offset]) \n"
                         "fbc.ps f29, 0x8(%[scale]) \n" 
                         "frcp.ps f29, f29 \n"
                         "fcvt.ps.pw f30, f30 \n"
                         "fmadd.ps f0, f0, f29, f30 \n"
                         "fcvt.pw.ps f0, f0 \n"
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ offset ] "r"(offset),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>

  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n" 
                         OPERATION_STEP3 
                         "fsrli.pi f1, f0, 0x8 \n"
                         "fxor.pi f29, f29, f29 \n"
                         "fcmov.ps f0, f1, f29, f0 \n"                        
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f1, f1, 0xff \n" 
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n" 
                         OPERATION_STEP3 
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n" 
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n" 
                         OPERATION_STEP3 
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n" 
                         "fandi.pi f1, f1, 0xff \n" 
                         OPERATION_STEP2
                         "fmadd.ps f0, f0, f29, f1 \n" 
                         OPERATION_STEP3 
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Add>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                         "fgh.ps  f0, f31(%[src1]) \n"  
                         "fcvt.ps.f16 f0, f0 \n"  
                         "fgh.ps  f1, f31(%[src2]) \n"  
                         "fcvt.ps.f16 f1, f1 \n"  
                         "fsub.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }

  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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

  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
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

  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n" 
                         OPERATION_STEP3 
                         "fsrli.pi f1, f0, 0x8 \n"
                         "fxor.pi f29, f29, f29 \n"
                         "fcmov.ps f0, f1, f29, f0 \n"                        
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f1, f1, 0xff \n" 
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n" 
                         OPERATION_STEP3 
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n" 
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n" 
                         OPERATION_STEP3 
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<int8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__(OPERATION_STEP1
                         "fandi.pi f0, f0, 0xff \n" 
                         "fandi.pi f1, f1, 0xff \n" 
                         OPERATION_STEP2
                         "fmsub.ps f0, f0, f29, f1 \n" 
                         OPERATION_STEP3 
                         "fsat8.pi f0, f0 \n"
                         "fscb.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Sub>::value && std::is_same<S1, Addresser<uint8_t>>::value && std::is_same<S2, Addresser<uint8_t>>::value && std::is_same<D, Addresser<uint8_t>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
                         : [ gatherValues ] "r"(gatherValues),
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                         "fgh.ps  f0, f31(%[src1]) \n"  
                         "fcvt.ps.f16 f0, f0 \n"  
                         "fgh.ps  f1, f31(%[src2]) \n"  
                         "fcvt.ps.f16 f1, f1 \n"  
                         "fmul.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }

  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Mul>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Mul>::value && (std::is_same<S1, Addresser<uint8_t>>::value || std::is_same<S2, Addresser<uint8_t>>::value) && std::is_same<D, Addresser<int8_t>>::value && !std::is_same<S1, Addresser<float>>::value && !std::is_same<S1, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Mul>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value, std::size_t>::type = 0>

  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                         "fgh.ps  f0, f31(%[src1]) \n"  
                         "fcvt.ps.f16 f0, f0 \n"  
                         "fgh.ps  f1, f31(%[src2]) \n"  
                         "fcvt.ps.f16 f1, f1 \n" 
                         "frcp.ps f1, f1 \n" 
                         "fmul.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }

  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Div>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Div>::value && (std::is_same<S1, Addresser<uint8_t>>::value || std::is_same<S2, Addresser<uint8_t>>::value) && std::is_same<D, Addresser<int8_t>>::value && !std::is_same<S1, Addresser<float>>::value && !std::is_same<S1, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Div>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value, std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n"
                         "flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
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
      typename U = opType, typename S = src1Type,
      typename std::enable_if<std::is_same<U, Div>::value &&
                                  !std::is_same<S, Addresser<int64_t>>::value,
                              std::size_t>::type = 0>
  void doOp(S &dst, const S &src1, const S &src2, uint64_t &d, uint64_t &s1,
            uint64_t &s2) {
    float inverted_op;
    getReciprocal(src2[s2], inverted_op);
    dst[d] = src1[s1] * inverted_op;
  }

   template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Max>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                         "fgh.ps  f0, f31(%[src1]) \n"  
                         "fcvt.ps.f16 f0, f0 \n"  
                         "fgh.ps  f1, f31(%[src2]) \n"  
                         "fcvt.ps.f16 f1, f1 \n"  
                         "fmax.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }

  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Max>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Max>::value && (std::is_same<S1, Addresser<uint8_t>>::value || std::is_same<S2, Addresser<uint8_t>>::value) && std::is_same<D, Addresser<int8_t>>::value && !std::is_same<S1, Addresser<float>>::value && !std::is_same<S1, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Max>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value, std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                         "fgh.ps  f0, f31(%[src1]) \n"  
                         "fcvt.ps.f16 f0, f0 \n"  
                         "fgh.ps  f1, f31(%[src2]) \n"  
                         "fcvt.ps.f16 f1, f1 \n"  
                         "fmin.ps f0, f0, f1 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f31", "memory");
  }

  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Min>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Min>::value && (std::is_same<S1, Addresser<uint8_t>>::value || std::is_same<S2, Addresser<uint8_t>>::value) && std::is_same<D, Addresser<int8_t>>::value && !std::is_same<S1, Addresser<float>>::value && !std::is_same<S1, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r"(dstAddr),
                           [ scale ] "r"(scale)
                         : "f0", "f1", "f29", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Min>::value && std::is_same<S1, Addresser<int8_t>>::value && std::is_same<S2, Addresser<int8_t>>::value && std::is_same<D, Addresser<int8_t>>::value, std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                         "fgh.ps  f0, f31(%[src1]) \n"  
                         "fcvt.ps.f16 f0, f0 \n"  
                         "fgh.ps  f1, f31(%[src2]) \n"  
                         "fcvt.ps.f16 f1, f1 \n"  
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
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f2", "f31", "memory");
  }

  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpEQ>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, float *scale, int32_t *offset) {
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                         "fgh.ps  f0, f31(%[src1]) \n"  
                         "fcvt.ps.f16 f0, f0 \n"  
                         "fgh.ps  f1, f31(%[src2]) \n"  
                         "fcvt.ps.f16 f1, f1 \n"  
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
                         "fsw.ps f2, 0x0(%[dst]) \n"
                         "mov.m.x m1, zero, 16 \n"
                         "mov.m.x m1, zero, 16 \n"
                         "maskand m0, m0, m1 \n"
                         "fsw.ps f2, -12(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src1 ] "r"(srcAddr1),
                           [ src2 ] "r"(srcAddr2),
                           [ dst ] "r" (dstAddr)
                         : "f0", "f1", "f2", "f31", "memory");
  }
  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, CmpLTE>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, float *scale, int32_t *offset) {
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, bool *dstAddr, float *scale, int32_t *offset) {
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
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
            typename std::enable_if<std::is_same<U, Pow>::value && std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    float half = 0.5;
    float minus2 = -2;
    __asm__ __volatile__("maskand m1, m0, m0 \n"
                         "flw.ps f28, 0x0(%[gatherValues])\n" 
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
                           [ gatherValues ] "r"(gatherValues)
                         : "f0", "f1", "f2", "f3", "f4", "f5", "f28", "f29", "f30", "f31", "memory");
  }

  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, Pow>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    float half = 0.5;
    float minus2 = -2;
    __asm__ __volatile__("maskand m1, m0, m0 \n"
                         "fbc.ps f31, 0x0(%[half]) \n"
                         "fbc.ps f30, 0x0(%[minus2]) \n"
                         "flw.ps f28, 0x0(%[gatherValues])\n" 
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
                           [ gatherValues ] "r"(gatherValues)
                         : "f0", "f1", "f2", "f3", "f4", "f5", "f26", "f27", "f28", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType, typename S1 = src1Type, typename S2 = src2Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Pow>::value && (std::is_same<S1, Addresser<uint8_t>>::value || std::is_same<S2, Addresser<uint8_t>>::value) && std::is_same<D, Addresser<int8_t>>::value && !std::is_same<S1, Addresser<float>>::value && !std::is_same<S1, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    float half = 0.5;
    float minus2 = -2;
    __asm__ __volatile__("maskand m1, m0, m0 \n"
                         "fbc.ps f31, 0x0(%[half]) \n"
                         "fbc.ps f30, 0x0(%[minus2]) \n"
                         "flw.ps f28, 0x0(%[gatherValues])\n" 
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
                           [ gatherValues ] "r"(gatherValues)
                         : "f0", "f1", "f2", "f3", "f4", "f5", "f27", "f28", "f29", "f30", "f31", "memory");
  }

  template <typename U = opType, typename S = src1Type, typename D = dstType,
            typename std::enable_if<std::is_same<U, Pow>::value && std::is_same<D, Addresser<uint8_t>>::value && !std::is_same<S, Addresser<float>>::value && !std::is_same<S, Addresser<float16>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr1, uintptr_t  srcAddr2, uintptr_t dstAddr, float *scale, int32_t *offset) {
    float half = 0.5;
    float minus2 = -2;
    __asm__ __volatile__("maskand m1, m0, m0 \n"
                         "fbc.ps f31, 0x0(%[half]) \n"
                         "fbc.ps f30, 0x0(%[minus2]) \n"
                         "flw.ps f28, 0x0(%[gatherValues])\n" 
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
                           [ gatherValues ] "r"(gatherValues)
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr, uintptr_t dstAddr, float *scale, int32_t *offset) {
    float log2e = M_1_LOG2E;
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                         "fbc.ps f30, 0x0(%[log2e]) \n"
                         "fgh.ps  f0, f31(%[src]) \n"  
                         "fcvt.ps.f16 f0, f0 \n"  
                         "flog.ps f0, f0 \n"
                         "fmul.ps f0, f0, f30 \n"
                         "fcvt.f16.ps f0, f0 \n"
                         "fsch.ps  f0, f31(%[dst]) \n"
                         :
                         : [ gatherValues ] "r"(gatherValues),
                           [ src ] "r"(srcAddr),
                           [ dst ] "r"(dstAddr),
                           [ log2e ] "r"(&log2e)
                         : "f0", "f30", "f31", "memory");
  }

  
  template <typename U = opType, typename S = src1Type,
            typename std::enable_if<std::is_same<U, ElementLog>::value && std::is_same<S, Addresser<float>>::value,
                                    std::size_t>::type = 0>
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr, uintptr_t dstAddr, float *scale, int32_t *offset) {
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
  void doOpVect(volatile int32_t *gatherValues, uintptr_t srcAddr, uintptr_t dstAddr, float *scale, int32_t *offset) {
    float log2e = M_1_LOG2E;
    __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
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
                         : [ gatherValues ] "r"(gatherValues),
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
};

template <typename srcType>
void addrCopy(uint64_t srcStartI, uint64_t srcEndI, uint64_t dstStartI,
              const Addresser<srcType> &tInput, Addresser<srcType> &tOutput) {
  // perform the copy
  auto val = tInput[0];
  for (uint64_t i = srcStartI, num = 0; i < srcEndI; i++, num++) {
    val = tInput[i];
    tOutput[dstStartI + num] = val;
  }
}

//===----------------------------------------------------------------------===//
//                       Quantization functions
//===----------------------------------------------------------------------===//

/// \returns the value \p in as clipped to the range of \p DestTy.
template <class SrcTy, class DestTy> DestTy dnn_lib::clip(SrcTy in) {
  static_assert(sizeof(SrcTy) >= sizeof(DestTy), "Invalid types");
  auto mx = std::numeric_limits<DestTy>::max();
  auto mn = std::numeric_limits<DestTy>::min();
  return std::max<SrcTy>(mn, std::min<SrcTy>(mx, in));
}

/// Converts floating point value to DestTy (int8 or int32) based on the
/// quantization parameters \p TQP.
template <class DestTy>
inline DestTy dnn_lib::quantize(float input, float scale, int32_t offset) {
  float invertedScale;
  fpReciprocalSingleElement(scale, invertedScale);
  float result = input * invertedScale + offset;
  return clip<int32_t, DestTy>((int32_t)nearbyintf(result));
}

// TODO Convert to int64_t
template <class SrcTy>
float dnn_lib::dequantize(SrcTy input, float scale, int32_t offset) {
  return scale * ((int32_t)input - offset);
}

/// Converts a quantized value (type eTy) to floating point based on the
/// quantization parameters \p scale and \p offset. If the input type is int8_t,
/// then an offset of 128 is added to convert to uint8_t.
template <class eTy>
inline float dnn_lib::dequantizeWithFloatOffset(eTy input, float scale,
                                                float offset) {
  uint8_t d = static_cast<uint8_t>(input);
  if (std::is_same<int8_t, eTy>::value) {
    d += 128;
  }
  return (d * scale) + offset;
}

int8_t dnn_lib::quantizeValInt8(float val, float scale, int32_t offset) {
  return quantize<int8_t>(val, scale, offset);
}

//===----------------------------------------------------------------------===//
//                       Convolution
//===----------------------------------------------------------------------===//
template <typename srcType>
void dnn_lib::fwdLibConvolutionInst(void *dstMatrix, void *dstMatrixDims,
                                    void *dstMatrixPitches, void *activations,
                                    void *activationsDims,
                                    void *activationsPitches, void *weights,
                                    void *weightsDims, void *weightPitches,
                                    void *bias, void *pkernels, void *pstrides,
                                    void *ppads, unsigned int group,
                                    float *scale, int32_t *offset) {

  // FIXME: going back to single thread until general case is solved with
  // multithread
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  float *tBias = (float *)bias;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides; // Jump between convolutions
  unsigned int *pads =
      (unsigned int *)ppads; // 0 added to avoid loss of dimensions

  assert(actIndex[3] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 &&
         "Output channels must be divisible by group.");
  size_t inCperG = actIndex[3] / group;
  size_t outCperG = dstIndex[3] / group;

  // For each input in the batch:
  for (size_t n = 0; n < actIndex[0]; n++) {

    // For each group of input channels:
    for (size_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (size_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pads[0]);
        for (size_t ax = 0; ax < dstIndex[1]; x += strides[0], ax++) {
          ssize_t y = -ssize_t(pads[1]);
          for (size_t ay = 0; ay < dstIndex[2]; y += strides[1], ay++) {

            // For each element in the convolution-filter:
            auto sum = tAInput[0];
            sum = 0;
            for (size_t fx = 0; fx < kernels[0]; fx++) {
              for (size_t fy = 0; fy < kernels[1]; fy++) {
                ssize_t ox = x + fx;
                ssize_t oy = y + fy;

                // Ignore index access below zero (this is due to padding). The
                // elegance of this should be improved
                if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
                    oy >= ssize_t(actIndex[2])) {
                  continue;
                }

                for (size_t fd = 0; fd < inCperG; fd++) {
                  auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                                     fy * weightPitch[2] + fd];
                  auto op2 =
                      tAInput[n * actPitch[0] + (size_t)ox * actPitch[1] +
                              (size_t)oy * actPitch[2] + g * inCperG + fd];
                  sum += op1 * op2;
                }
              }
            }
            int64_t addr = n * dstPitch[0] + (size_t)ax * dstPitch[1] +
                           (size_t)ay * dstPitch[2] + d;
            sum += tBias[d];
            tOutput[addr] = sum;
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

template <typename srcType>
void dnn_lib::fwdLibConvolutionInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    void *pkernels, void *pstrides, void *ppads, unsigned int group,
    float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  float *tBias = (float *)bias;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides; // Jump between convols
  unsigned int *pads = (unsigned int *)ppads; // 0 added to avoid loss of dims


  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  assert(actIndex[3] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 &&
         "Output channels must be divisible by group.");
  unsigned int inCperG = actIndex[3] / group;
  unsigned int outCperG = dstIndex[3] / group;

  unsigned int eDstPitch[5] = {dstPitch[0], dstPitch[1], dstPitch[2], outCperG,
                               1};
    
  unsigned int eDstIndex[5] = {dstIndex[0], dstIndex[1], dstIndex[2], group,
                               outCperG};

  unsigned int coord[5], k;
  getNonPaddingCoordinates(coord, initialAddr, 5, eDstPitch, eDstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, d;
  while ((offsetOut < posMax) && !done) {
    x = coord[1] * strides[0] - ssize_t(pads[0]); // least x coordinate in kernel
    y = coord[2] * strides[1] - ssize_t(pads[1]); // least y coordinate in kernel
    d = coord[3] * outCperG + coord[4];           // depth in kernel

    auto sum = tAInput[0];                        //Same type as tAInput[]
    sum = tBias[d];// 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }
    //sum += tBias[d];
    tOutput[offsetOut] = sum;

    done = getOffsets(5, coord, offsetOut, eDstIndex, eDstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

// template <typename srcType>
// void dnn_lib::fwdLibConvolutionInstThreaded(void *dstMatrix, void
// *dstMatrixDims,
//                                    void *dstMatrixPitches, void *activations,
//                                    void *activationsDims,
//                                    void *activationsPitches, void *weights,
//                                    void *weightsDims, void *weightPitches,
//                                    void *bias, void *pkernels, void
//                                    *pstrides, void *ppads, unsigned int
//                                    group, float *scale, int32_t *offset) {
//
//  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
//  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
//  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
//  float *tBias = (float *)bias;
//
//  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
//  unsigned int *actIndex = (unsigned int *)activationsDims;
//  unsigned int *weightIndex = (unsigned int *)weightsDims;
//
//  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
//  unsigned int *actPitch = (unsigned int *)activationsPitches;
//  unsigned int *weightPitch = (unsigned int *)weightPitches;
//
//  unsigned int *kernels = (unsigned int *)pkernels;
//  unsigned int *strides = (unsigned int *)pstrides;   // Jump between
//  convolutions unsigned int *pads = (unsigned int *)ppads;
//  // 0 added to avoid loss of dimensions
//
//  unsigned int minionId = get_minion_id();
//  unsigned int numElemsKernel = kernels[0]*kernels[1];
//  unsigned int minionsperkernel = 1;
//  int level = -1;
//  while (minionsperkernel < numElemsKernel) {
//    minionsperkernel*= 2;
//    ++level;
//  }
//  unsigned int numKernels = 1024/minionsperkernel;
//  unsigned int kernel_id = minionId/minionsperkernel;
//  unsigned int kernel_minionId = minionId - kernel_id*minionsperkernel;
//  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];
//  unsigned int cll = 64/sizeof(srcType);
//  unsigned int ncl = (numElemsDst - 1)/cll + 1; //amount of cache lines
//  unsigned int kcl = (ncl-1)/numKernels + 1; //Amount of cache lines to do for
//  the kernel unsigned int initialAddr = kcl*cll*kernel_id;
//
//  assert((actIndex[3] % group == 0) &&
//         "Input channels must be divisible by group.");
//  assert((dstIndex[3] % group == 0) &&
//         "Output channels must be divisible by group.");
//  unsigned int inCperG = actIndex[3] / group;
//  unsigned int outCperG = dstIndex[3] / group;
//
//  // We treat groups as a new dimension, with their corresponding pitches,
//  assuming that there's no padding between groups unsigned int eDstPitch[5] =
//  {dstPitch[0], dstPitch[1], dstPitch[2], outCperG, 1}; unsigned int
//  eDstIndex[5] = {dstIndex[0], dstIndex[1], dstIndex[2], group, outCperG};
//
//  unsigned int coord[5] = {0,0,0,0,0};
//  unsigned int rm = initialAddr;
//  for (unsigned int i = 0; i < 5; i++) {
//    coord[i] = rm/eDstPitch[i];
//    rm = rm-coord[i]*eDstPitch[i];
//  }
//
//  unsigned int k = 5; //If it is a padding position we compute next useful
//  position for (unsigned int j = 4; j > 0; j--) {
//    if (coord[j] >= eDstIndex[j]) {
//      coord[j-1]++;
//      k = j;
//    }
//  }
//  for (unsigned int j = k; j < 5; j++) coord[j] = 0;
//
//  int64_t offsetOut = 0;
//  for (int i = 0; i < 5; i++) offsetOut += coord[i]*eDstPitch[i];
//  if (offsetOut >= numElemsDst) return;
//
//  unsigned int maxRead = kcl*cll;
//  unsigned int posMax = maxRead + offsetOut;
//  ssize_t x, y, dx, dy, x_ker, y_ker;
//  x_ker = kernel_minionId/kernels[1];
//  y_ker = kernel_minionId - x_ker*kernels[1];
//  dx = x_ker - ssize_t(pads[0]);
//  dy = y_ker - ssize_t(pads[1]);
//
//  bool done = false;
//  while(!done) {
//
//    x = coord[1]*strides[0] + dx;
//    y = coord[2]*strides[1] + dy;
//
//    srcType results[outCperG];
//    if ((x >= 0) && (y >= 0) && (x < ssize_t(actIndex[1])) &&
//        (y < ssize_t(actIndex[2])) && (kernel_minionId < numElemsKernel)) {
//      for (int k = 0; k < outCperG; k++) {
//        auto sum = tAInput[0];
//        sum = 0;
//        int filter = coord[3]*outCperG + k;
//        for (int z = 0; z < inCperG; z++) {
//          auto op1 = tWInput[filter * weightPitch[0] + x_ker * weightPitch[1]
//          +
//                             y_ker * weightPitch[2] + z];
//          auto op2 = tAInput[coord[0] * actPitch[0] + x * actPitch[1] +
//                             y * actPitch[2] + coord[3] * inCperG + z];
//          sum += op1 * op2;
//        } 
//        results[k] = sum;
//      }
//    }
//    for (int k = 0; k < outCperG; k++) {
//      for (int i = 0; i <= level; i++) {
//        results[k] = tensor_reduce_float(results[k], 0x0, 1, i, 0x3);
//      }
//    }
//    if (kernel_minionId == 0) {
//      for (int k = 0; k < outCperG; k++) {
//        int filter = coord[3]*outCperG + k;
//        tOutput[offsetOut + k] = (results[k] + tBias[filter]);
//      }
//    }
//
//    done = getOffsets(4, coord, offsetOut, eDstIndex, eDstPitch);
//    if (offsetOut >= posMax) break;
//  }
//}

template <typename srcType, typename std::enable_if<std::is_same<
                            srcType, float>::value, std::size_t>::type = 0>
void convolutionOp (void *activations, void *weights, unsigned int *coord, 
                    unsigned int *actPitch, unsigned int *weightPitch, 
                    unsigned int *actIndex, unsigned int *kernels,
                    unsigned int inCperG, float &sum, int32_t mask, ssize_t x,
                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
  int dist;
  ssize_t fx, fy, ox, oy;
  fx = fy = 0;
  unsigned int *actAddr = (unsigned int *) activations; 
  unsigned int *weightAddr = (unsigned int *) weights;
  actAddr += coord[0] * actPitch[0] + x * actPitch[1] + y * actPitch[2] +
            coord[3] * inCperG;
  weightAddr += d * weightPitch[0];
  __asm__ __volatile__(
    "fxor.pi  f0, f0, f0\n"                         // f0 to zeros      
    "mov.m.x  m0, zero, 0xff\n"                     // m0 to ones
    "mov.m.x  m1, %[mask], 0\n"                     // m1 the auxiliar mask
    "1:\n"                                          // for (size_t fx = 0; fx < kernels[0]; fx++) {
    "beq      %[fy], zero, 2f\n"
    "mul      %[fy], %[kernels1], %[actPitch2]\n"
    "sub      %[actAddr], %[actAddr], %[fy]\n"
    "mul      %[fy], %[kernels1], %[weightPitch2]\n"
    "sub      %[weightAddr], %[weightAddr], %[fy]\n"
    "addi     %[fy], zero, 0\n"                     
    "2:\n"                                            // for (size_t fy = 0; fy < kernels[1]; fy++) {
    "addi     %[dist], %[inCperG], 0\n"                // dist = inCperG

    "add      %[oy], %[y], %[fy]\n"                     // oy = y + fy
    "add      %[ox], %[x], %[fx]\n"                     // ox = x + fx
    
    "blt      %[ox], zero, 5f\n"                        // if (ox < 0) continue
    "blt      %[oy], zero, 5f\n"                        // if (oy < 0) continue
    "ble      %[actIndex1], %[ox], 5f\n"                // if (actIndex[1] <= ox) continue
    "ble      %[actIndex2], %[oy], 5f\n"                // if (actIndex[2] <= oy) continue
    
    "addi     t0, zero, 8\n"                            // t0 = 8
    "ble      %[dist], t0, 4f\n"                        // if dist <= 8 go to 4
  
    "mov.m.x  m0, zero, 0xff\n"
    "3:\n"                                              // while (8 < dist) {
    "flw.ps   f1, 0x0(%[actAddr])\n"                      // actAddr -> f1
    "flw.ps   f2, 0x0(%[weightAddr])\n"                   // weightaddr -> f2
    "fmadd.ps f0, f1, f2, f0\n"                           // f0 = (f1 * f2) + f0
    "addi     %[actAddr], %[actAddr], 32\n"               // actAddr += 32
    "addi     %[weightAddr], %[weightAddr], 32\n"         // weightAddr += 32
    "addi     %[dist], %[dist], -8\n"                     // dist -= 8
    "blt      t0, %[dist], 3b\n"                        // }
    
    "4:\n"
    "maskand  m0, m0, m1\n"                             // put mask on
    "flw.ps   f1, 0x0(%[actAddr])\n"                    // actAddr -> f1
    "flw.ps   f2, 0x0(%[weightAddr])\n"                 // weightaddr -> f2
    "fmadd.ps f0, f1, f2, f0\n"                         // f0 = (f1 * f2) + f0
    "sub      %[dist], %[inCperG], %[dist]\n"           // dist = inCperG - dist
    "slli     %[dist], %[dist], 2\n"                    // dist = dist * 4
    "sub      %[actAddr], %[actAddr], %[dist]\n"        // actAddr = actAddr - dist
    "sub      %[weightAddr], %[weightAddr], %[dist]\n"  // actAddr = actAddr - dist

    "5:\n"
    "addi     %[fy], %[fy], 1\n"                        // fy++
    "add     %[actAddr], %[actPitch2], %[actAddr]\n"   // actAddr = actAddr + actPitch[2]
    "add     %[weightAddr], %[weightPitch2], %[weightAddr]\n"
    "blt      %[fy], %[kernels1], 2b\n"               // Closing fy for }

    "addi     %[fx], %[fx], 1\n"                      // fx++
    
    "add     %[actAddr], %[actPitch1], %[actAddr]\n" // actAddr = actAddr + actPitch[1]
    "add     %[weightAddr], %[weightPitch1], %[weightAddr]\n"
    "blt      %[fx], %[kernels0], 1b\n"             // Closing fx for{}
    
    "mov.m.x   m0, zero, 0xff\n" 
    "fswizz.ps f1, f0, 0xe\n"
    "fadd.ps   f0, f0, f1\n"
    "fswizz.ps f1, f0, 0x1\n"
    "fadd.ps   f0, f0, f1\n"
    "fmvs.x.ps t0, f0, 0x4\n"
    "fmv.w.x   f31, t0\n"
    "fadd.s    f31, f31, f0\n"   

    "fmv.w.x   f0, %[sum]\n"
    "fadd.s    f31, f31, f0\n" 
    "fmv.x.w   %[sum], f31\n"
    
    : [ weightAddr ] "+r" (weightAddr), 
      [ actAddr ] "+r" (actAddr),       
      [ dist ] "+r" (dist),             
      [ sum ] "+r" (sum),               
      [ ox ] "+r" (ox),                 
      [ oy ] "+r" (oy),                 
      [ fy ] "+r" (fy),   
      [ fx ] "+r" (fx)  
    : [ weightPitch1 ] "r" (weightPitch[1] * 4),  
      [ weightPitch2 ] "r" (weightPitch[2] * 4),  
      [ actIndex1 ] "r" (actIndex[1]),           
      [ actIndex2 ] "r" (actIndex[2]),           
      [ actPitch1 ] "r" (actPitch[1] * 4),       
      [ actPitch2 ] "r" (actPitch[2] * 4),       
      [ kernels0 ] "r" (kernels[0]),             
      [ kernels1 ] "r" (kernels[1]),             
      [ inCperG ] "r" (inCperG),                 
      [ mask ] "r" (mask),                        
      [ x ] "r" (x),                              
      [ y ] "r" (y)                               
    : "memory", "f0", "f1", "f2", "f31", "t0", "t1");
  return;
}

template <typename srcType, typename std::enable_if<(!std::is_same<
                            srcType, float>::value) /*&& (!std::is_same<
                            srcType, float16>::value) && (!std::is_same<
                            srcType, int8_t>::value)*/, std::size_t>::type = 0>
void convolutionOp (void *activations, void *weights, unsigned int *coord, 
                    unsigned int *actPitch, unsigned int *weightPitch, 
                    unsigned int *actIndex, unsigned int *kernels,
                    unsigned int inCperG, float &sum, int32_t mask, ssize_t x,
                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }  
  return; //TODO return error.
}

template <typename srcType, typename std::enable_if<std::is_same<
                            srcType, float16>::value, std::size_t>::type = 0>
void convolutionOp (void *activations, void *weights, unsigned int *coord, 
                    unsigned int *actPitch, unsigned int *weightPitch, 
                    unsigned int *actIndex, unsigned int *kernels,
                    unsigned int inCperG, float16 &sum, int32_t mask, ssize_t x,
                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
  int dist;
  ssize_t fx, fy, ox, oy;
  fx = fy = 0;
  uint16_t *actAddr = (uint16_t *) activations; 
  uint16_t *weightAddr = (uint16_t *) weights;
  actAddr += coord[0] * actPitch[0] + x * actPitch[1] + y * actPitch[2] +
            coord[3] * inCperG;
  weightAddr += d * weightPitch[0];
  unsigned int gatherValues[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };
  __asm__ __volatile__(
    "flw.ps f16, 0x0(%[gatherValues])\n"
    "fxor.pi  f0, f0, f0\n"                         // f0 to zeros      
    "mov.m.x  m0, zero, 0xff\n"                     // m0 to ones
    "mov.m.x  m1, %[mask], 0\n"                     // m1 the auxiliar mask
    "1:\n"                                          // for (size_t fx = 0; fx < kernels[0]; fx++) {
    "beq      %[fy], zero, 2f\n"
    "mul      %[fy], %[kernels1], %[actPitch2]\n"
    "sub      %[actAddr], %[actAddr], %[fy]\n"
    "mul      %[fy], %[kernels1], %[weightPitch2]\n"
    "sub      %[weightAddr], %[weightAddr], %[fy]\n"
    "addi     %[fy], zero, 0\n"                     
    "2:\n"                                            // for (size_t fy = 0; fy < kernels[1]; fy++) {
    "addi     %[dist], %[inCperG], 0\n"                // dist = inCperG

    "add      %[oy], %[y], %[fy]\n"                     // oy = y + fy
    "add      %[ox], %[x], %[fx]\n"                     // ox = x + fx
    
    "blt      %[ox], zero, 5f\n"                        // if (ox < 0) continue
    "blt      %[oy], zero, 5f\n"                        // if (oy < 0) continue
    "ble      %[actIndex1], %[ox], 5f\n"                // if (actIndex[1] <= ox) continue
    "ble      %[actIndex2], %[oy], 5f\n"                // if (actIndex[2] <= oy) continue
    
    "addi     t0, zero, 8\n"                            // t0 = 8
    "ble      %[dist], t0, 4f\n"                        // if dist <= 8 go to 4
  
    "mov.m.x  m0, zero, 0xff\n"
    "3:\n"                                              // while (8 < dist) {
    "fgh.ps   f1, f16(%[actAddr])\n"                      // actAddr -> f1
    "fcvt.ps.f16 f1, f1\n"
    "fgh.ps   f2, f16(%[weightAddr])\n"                   // weightaddr -> f2
    "fcvt.ps.f16 f2, f2\n"
    "fmadd.ps f0, f1, f2, f0\n"                           // f0 = (f1 * f2) + f0
    "addi     %[actAddr], %[actAddr], 16\n"               // actAddr += 16
    "addi     %[weightAddr], %[weightAddr], 16\n"         // weightAddr += 16
    "addi     %[dist], %[dist], -8\n"                     // dist -= 8
    "blt      t0, %[dist], 3b\n"                        // }
    
    "4:\n"
    "maskand  m0, m0, m1\n"                             // put mask on
    "fgh.ps   f1, f16(%[actAddr])\n"                    // actAddr -> f1
    "fcvt.ps.f16 f1, f1\n"
    "fgh.ps   f2, f16(%[weightAddr])\n"                 // weightaddr -> f2
    "fcvt.ps.f16 f2, f2\n"
    "fmadd.ps f0, f1, f2, f0\n"                         // f0 = (f1 * f2) + f0
    "sub      %[dist], %[inCperG], %[dist]\n"           // dist = inCperG - dist
    "slli     %[dist], %[dist], 1\n"                    // dist = dist * 2
    "sub      %[actAddr], %[actAddr], %[dist]\n"        // actAddr = actAddr - dist
    "sub      %[weightAddr], %[weightAddr], %[dist]\n"  // actAddr = actAddr - dist

    "5:\n"
    "addi     %[fy], %[fy], 1\n"                        // fy++
    "add      %[actAddr], %[actPitch2], %[actAddr]\n"   // actAddr = actAddr + actPitch[2]
    "add      %[weightAddr], %[weightPitch2], %[weightAddr]\n"
    "blt      %[fy], %[kernels1], 2b\n"               // Closing fy for{}

    "addi     %[fx], %[fx], 1\n"                      // fx++
    
    "add      %[actAddr], %[actPitch1], %[actAddr]\n" // actAddr = actAddr + actPitch[1]
    "add      %[weightAddr], %[weightPitch1], %[weightAddr]\n"
    "blt      %[fx], %[kernels0], 1b\n"             // Closing fx for{}
    
    "mov.m.x   m0, zero, 0xff\n" 
    "fswizz.ps f1, f0, 0xe\n"
    "fadd.ps   f0, f0, f1\n"
    "fswizz.ps f1, f0, 0x1\n"
    "fadd.ps   f0, f0, f1\n"
    "fmvs.x.ps t0, f0, 0x4\n"
    "fmv.w.x   f31, t0\n"
    "fadd.s    f31, f31, f0\n"   
    "fmv.w.x   f0, %[sum]\n"
    "fadd.s    f31, f31, f0\n" 
    "fmv.x.w   %[sum], f31\n"
    
    : [ weightAddr ] "+r" (weightAddr), 
      [ actAddr ] "+r" (actAddr),       
      [ dist ] "+r" (dist),             
      [ sum ] "+r" (sum),               
      [ ox ] "+r" (ox),                 
      [ oy ] "+r" (oy),                 
      [ fy ] "+r" (fy),   
      [ fx ] "+r" (fx)  
    : [ weightPitch1 ] "r" (weightPitch[1] * 2),  
      [ weightPitch2 ] "r" (weightPitch[2] * 2),  
      [ gatherValues ] "r" (gatherValues),        
      [ actPitch1 ] "r" (actPitch[1] * 2),       
      [ actPitch2 ] "r" (actPitch[2] * 2),       
      [ actIndex1 ] "r" (actIndex[1]),           
      [ actIndex2 ] "r" (actIndex[2]),  
      [ kernels0 ] "r" (kernels[0]),             
      [ kernels1 ] "r" (kernels[1]),             
      [ inCperG ] "r" (inCperG),                 
      [ mask ] "r" (mask),                        
      [ x ] "r" (x),                              
      [ y ] "r" (y)                               
    : "memory", "f0", "f1", "f2", "f31", "t0", "t1");
  return;
}

//template <typename srcType, typename std::enable_if<std::is_same<
//                            srcType, int8_t>::value, std::size_t>::type = 0>
//void convolutionOp (void *activations, void *weights, unsigned int *coord, 
//                    unsigned int *actPitch, unsigned int *weightPitch, 
//                    unsigned int *actIndex, unsigned int *kernels,
//                    unsigned int inCperG, srcType &sum, int32_t mask, ssize_t x,
//                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
//
//  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
//  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
//  
//  for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
//      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
//        ssize_t ox = x + fx;
//        ssize_t oy = y + fy;
//
//        // Ignore index access below zero (this is due to padding).
//        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
//            oy >= ssize_t(actIndex[2])) {
//          continue;
//        }
//        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
//          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
//                             fy * weightPitch[2] + fd];
//          auto op2 =
//              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
//                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
//          sum += op1 * op2;
//        }
//      }
//    }
//  return; //TODO a version of int8_t is needed.  
//}

template <typename srcType, typename std::enable_if</*(!std::is_same<
                            srcType, float>::value) && */(!std::is_same<
                            srcType, float16>::value) /*&& (!std::is_same<
                            srcType, int8_t>::value)*/, std::size_t>::type = 0>
void convolutionOp (void *activations, void *weights, unsigned int *coord, 
                    unsigned int *actPitch, unsigned int *weightPitch, 
                    unsigned int *actIndex, unsigned int *kernels,
                    unsigned int inCperG, float16 &sum, int32_t mask, ssize_t x,
                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }  
  return; //TODO return error.
}

template <typename srcType>
void convolutionOp (void *activations, void *weights, unsigned int *coord, 
                    unsigned int *actPitch, unsigned int *weightPitch, 
                    unsigned int *actIndex, unsigned int *kernels,
                    unsigned int inCperG, int32_t &sum, int32_t mask, ssize_t x,
                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }  
  return; //TODO return error.
}



template <typename srcType>
void dnn_lib::fwdLibConvolutionInstVectorized(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    void *pkernels, void *pstrides, void *ppads, unsigned int group,
    float *scale, int32_t *offset, uint64_t flags) {

  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;
  float *tBias = (float *)bias;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides; // Jump between convols
  unsigned int *pads = (unsigned int *)ppads; // 0 added to avoid loss of dims


  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  assert(actIndex[3] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 &&
         "Output channels must be divisible by group.");
  unsigned int inCperG = actIndex[3] / group;
  unsigned int outCperG = dstIndex[3] / group;

  unsigned int eDstPitch[5] = {dstPitch[0], dstPitch[1], dstPitch[2], outCperG,
                               1};
    
  unsigned int eDstIndex[5] = {dstIndex[0], dstIndex[1], dstIndex[2], group,
                               outCperG};

  unsigned int coord[5], k;
  getNonPaddingCoordinates(coord, initialAddr, 5, eDstPitch, eDstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, d;
  volatile int32_t mask = (1 << (((inCperG - 1) & 0x7)  + 1)) - 1;
  while ((offsetOut < posMax) && !done) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);
    d = coord[3] * outCperG + coord[4];

    auto sum = tBias[d];
    volatile int dist;
    volatile unsigned int *actAddr = (unsigned int *) activations; 
    volatile unsigned int *weightAddr = (unsigned int *) weights;
    convolutionOp <srcType> (activations, weights, coord, actPitch, weightPitch,
                             actIndex, kernels, inCperG, sum, mask, x, y, d,
                             scale, offset);
    tOutput[offsetOut] = sum;

    done = getOffsets(5, coord, offsetOut, eDstIndex, eDstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}


template <typename srcType>
void dnn_lib::fwdLibConvolution3DInst(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    void *pkernels, void *pstrides, void *ppads, unsigned int group,
    float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  float *tBias = (float *)bias;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides;
  unsigned int *pads = (unsigned int *)ppads;

  assert(actIndex[4] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[4] % group == 0 &&
         "Output channels must be divisible by group.");
  size_t inCperG = actIndex[4] / group;
  size_t outCperG = dstIndex[4] / group;

  // For each input in the batch:
  for (size_t n = 0; n < actIndex[0]; n++) {

    // For each group of input channels:
    for (size_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (size_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pads[0]);
        for (size_t ax = 0; ax < dstIndex[1]; x += strides[0], ax++) {
          ssize_t y = -ssize_t(pads[1]);
          for (size_t ay = 0; ay < dstIndex[2]; y += strides[1], ay++) {
            ssize_t z = -ssize_t(pads[2]);
            for (size_t az = 0; az < dstIndex[3]; z += strides[2], az++) {

              // For each element in the 3Dconvolution-filter:
              auto sum = tAInput[0];
              sum = 0;
              for (size_t fx = 0; fx < kernels[0]; fx++) {
                for (size_t fy = 0; fy < kernels[1]; fy++) {
                  for (size_t fz = 0; fz < kernels[2]; fz++) {
                    ssize_t ox = x + fx;
                    ssize_t oy = y + fy;
                    ssize_t oz = z + fz;

                    // Ignore index access below zero (this is due to padding).
                    if (ox < 0 || oy < 0 || oz < 0 ||
                        ox >= ssize_t(actIndex[1]) ||
                        oy >= ssize_t(actIndex[2]) ||
                        oz >= ssize_t(actIndex[3])) {
                      continue;
                    }
                    for (size_t fd = 0; fd < inCperG; fd++) {
                      auto op1 =
                          tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                                  fy * weightPitch[2] + fz * weightPitch[3] +
                                  fd];
                      auto op2 =
                          tAInput[n * actPitch[0] + (size_t)ox * actPitch[1] +
                                  (size_t)oy * actPitch[2] +
                                  (size_t)oz * actPitch[3] + g * inCperG + fd];
                      sum += op1 * op2;
                    }
                  }
                }
              }
              int64_t addr = n * dstPitch[0] + (size_t)ax * dstPitch[1] +
                             (size_t)ay * dstPitch[2] +
                             (size_t)az * dstPitch[3] + d;
              sum += tBias[d];
              tOutput[addr] = sum;
            } // D
          }   // W
        }     // H
      }       // C
    }         // G
  }           // N
}

template <typename srcType>
void dnn_lib::fwdLibConvolution3DInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    void *pkernels, void *pstrides, void *ppads, unsigned int group,
    float *scale, int32_t *offset, uint64_t flags) {
 
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  float *tBias = (float *)bias;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides;
  unsigned int *pads = (unsigned int *)ppads;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;
  assert(actIndex[4] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[4] % group == 0 &&
         "Output channels must be divisible by group.");
  unsigned int inCperG = actIndex[4] / group;
  unsigned int outCperG = dstIndex[4] / group;

  unsigned int eDstPitch[6] = {dstPitch[0], dstPitch[1], dstPitch[2],
                               dstPitch[3], outCperG,    1};
  unsigned int eDstIndex[6] = {dstIndex[0], dstIndex[1], dstIndex[2],
                               dstIndex[3], group,       outCperG};

  unsigned int coord[6] = {0, 0, 0, 0, 0, 0};
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 6, eDstPitch, eDstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, z, d;
  while (!done && (offsetOut < posMax)) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);
    z = coord[3] * strides[2] - ssize_t(pads[2]);
    d = coord[4] * outCperG + coord[5];

    auto sum = tAInput[0];
    sum = 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {
      for (size_t fy = 0; fy < kernels[1]; fy++) {
        for (size_t fz = 0; fz < kernels[2]; fz++) {
          ssize_t ox = x + fx;
          ssize_t oy = y + fy;
          ssize_t oz = z + fz;

          // Ignore index access below zero (this is due to padding).
          if (ox < 0 || oy < 0 || oz < 0 || ox >= ssize_t(actIndex[1]) ||
              oy >= ssize_t(actIndex[2]) || oz >= ssize_t(actIndex[3])) {
            continue;
          }
          for (size_t fd = 0; fd < inCperG; fd++) {
            auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                               fy * weightPitch[2] + fz * weightPitch[3] + fd];
            auto op2 =
                tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                        (size_t)oy * actPitch[2] + (size_t)oz * actPitch[3] +
                        coord[4] * inCperG + fd];
            sum += op1 * op2;
          }
        }
      }
    }
    sum += tBias[d];
    tOutput[offsetOut] = sum;

    done = getOffsets(6, coord, offsetOut, eDstIndex, eDstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

//===----------------------------------------------------------------------===//
//                       Pooling
//===----------------------------------------------------------------------===//
template <typename srcType>
void dnn_lib::fwdLibMaxPoolInst(bool XY, void *dstMatrix, void *dstMatrixDims,
                                void *dstMatrixPitches, void *dst2Matrix,
                                void *dst2MatrixDims, void *dst2MatrixPitches,
                                void *activations, void *activationsDims,
                                void *activationsPitches, void *pkernels,
                                void *pstrides, void *ppads, float *scale,
                                int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  uint64_t *tOutput2 = (uint64_t *)dst2Matrix;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *dst2Index = (unsigned int *)dst2MatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *dst2Pitch = (unsigned int *)dst2MatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides;
  unsigned int *pads = (unsigned int *)ppads;

  // For each input in the batch:
  for (size_t n = 0; n < dstIndex[0]; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < actIndex[3]; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pads[0]);
      for (size_t ax = 0; ax < dstIndex[1]; x += strides[0], ax++) {
        ssize_t y = -ssize_t(pads[1]);
        for (size_t ay = 0; ay < dstIndex[2]; y += strides[1], ay++) {
          size_t maxX = x;
          size_t maxY = y;

          bool first = true;
          auto max_value = tAInput[0];
          max_value = 0;

          for (size_t fx = 0; fx < kernels[0]; fx++) {
            for (size_t fy = 0; fy < kernels[1]; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
                  oy >= ssize_t(actIndex[2])) {
                continue;
              }

              auto val = tAInput[n * actPitch[0] + (size_t)ox * actPitch[1] +
                                 (size_t)oy * actPitch[2] + z];
              if (first || (val >= max_value)) {
                first = false;
                max_value = val;
                maxX = ox;
                maxY = oy;
              }
            }
          }

          int64_t dstAddr = n * dstPitch[0] + (size_t)ax * dstPitch[1] +
                            (size_t)ay * dstPitch[2] + z;
          tOutput[dstAddr] = max_value;

          if (XY) {
            int64_t dst2Addr = n * dst2Pitch[0] + (size_t)ax * dst2Pitch[1] +
                               (size_t)ay * dst2Pitch[2] + z * dst2Pitch[3];
            tOutput2[dst2Addr] = (long long)maxX;
            tOutput2[dst2Addr + dst2Pitch[4]] = (long long)maxY;
          }
        } // W
      }   // H
    }     // C
  }       // N
}

template <typename srcType, typename dstType>
void dnn_lib::fwdLibMaxPoolInstThreaded(
    bool XY, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *pkernels, void *pstrides, void *ppads, float *scale, int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<dstType> tOutput(dstMatrix, scale[1], offset[1]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  uint64_t *tOutput2 = (uint64_t *)dst2Matrix;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *dst2Index = (unsigned int *)dst2MatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *dst2Pitch = (unsigned int *)dst2MatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides;
  unsigned int *pads = (unsigned int *)ppads;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[4] = {0, 0, 0, 0};
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 4, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y;
  while (!done && (offsetOut < posMax)) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);
    size_t maxX = x;
    size_t maxY = y;

    bool first = true;
    auto max_value = tAInput[0];
    max_value = 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {
      for (size_t fy = 0; fy < kernels[1]; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }

        auto val = tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                           (size_t)oy * actPitch[2] + coord[3] * actPitch[3]];
        if (first || (val >= max_value)) {
          first = false;
          max_value = val;
          maxX = ox;
          maxY = oy;
        }
      }
    }

    tOutput[offsetOut] = max_value;

    if (XY) {
      int64_t dst2Addr = coord[0] * dst2Pitch[0] + coord[1] * dst2Pitch[1] +
                         coord[2] * dst2Pitch[2] + coord[3] * dst2Pitch[3];
      tOutput2[dst2Addr] = (long long)maxX;
      tOutput2[dst2Addr + dst2Pitch[4]] = (long long)maxY;
    }
    done = getOffsets(4, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibAvgPoolInst(void *dstMatrix, void *dstMatrixDims,
                                void *dstMatrixPitches, void *activations,
                                void *activationsDims, void *activationsPitches,
                                void *pkernels, void *pstrides, void *ppads,
                                float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides;
  unsigned int *pads = (unsigned int *)ppads;

  float filterArea = kernels[0] * kernels[1];
  float invFilter;
  fpReciprocalSingleElement(filterArea, invFilter);

  // For each input in the batch:
  for (size_t n = 0; n < dstIndex[0]; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < actIndex[3]; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pads[0]);
      for (size_t ax = 0; ax < dstIndex[1]; x += strides[0], ax++) {
        ssize_t y = -ssize_t(pads[1]);
        for (size_t ay = 0; ay < dstIndex[2]; y += strides[1], ay++) {
          auto sum = tAInput[0];
          sum = 0;

          for (size_t fx = 0; fx < kernels[0]; fx++) {
            for (size_t fy = 0; fy < kernels[1]; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
                  oy >= ssize_t(actIndex[2])) {
                continue;
              }

              sum += tAInput[n * actPitch[0] + (size_t)ox * actPitch[1] +
                             (size_t)oy * actPitch[2] + z];
            }
          }
          float tmp = sum;
          tmp *= invFilter;
          sum = tmp;
          tOutput[n * dstPitch[0] + (size_t)ax * dstPitch[1] +
                  (size_t)ay * dstPitch[2] + z] = sum;
        } // W
      }   // H
    }     // C
  }       // N
}

// template <typename srcType>
// void dnn_lib::fwdLibAvgPoolInst_Copy(void *dstMatrix, void *dstMatrixDims,
//                                      void *dstMatrixPitches, void *activations,
//                                      void *activationsDims, void
//                                      *activationsPitches, void *pkernels, void
//                                      *pstrides, void *ppads, float *scale, int32_t
//                                      *offset) {
//  Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
//  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
//
//  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
//  unsigned int *actIndex = (unsigned int *)activationsDims;
//
//  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
//  unsigned int *actPitch = (unsigned int *)activationsPitches;
//
//  unsigned int *kernels = (unsigned int *)pkernels;
//  unsigned int *strides = (unsigned int *)pstrides;
//  unsigned int *pads = (unsigned int *)ppads;
//
//  float filterArea = kernels[0] * kernels[1];
//  float invFilter;
//  fpReciprocalSingleElement(filterArea, invFilter);
//  unsigned int minionId = get_minion_id();
//  unsigned int numElemsKernel = kernels[0]*kernels[1];
//  unsigned int minionsperkernel = 1;
//  int level = -1;
//  while (minionsperkernel < numElemsKernel) {
//    minionsperkernel*= 2;
//    ++level;
//  }
//  unsigned int numKernels = activeMinions/minionsperkernel;
//  unsigned int kernel_id = minionId/minionsperkernel;
//  unsigned int kernel_minionId = minionId - kernel_id*minionsperkernel;
//  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];
//  unsigned int cll = 64/sizeof(srcType);
//  unsigned int ncl = (numElemsDst - 1)/cll + 1; //Amount of cache lines
//  unsigned int kcl = (ncl-1)/numKernels + 1; //Amount of cache lines to do for
//  each kernel unsigned int initialAddr = kcl*cll*kernel_id; unsigned int
//  maxRead = kcl*cll; unsigned int posMax = maxRead + initialAddr;
//
//  if (initialAddr >= numElemsDst) return;
//
//  unsigned int offsetOut = initialAddr;
//
//  long unsigned int coord[4] = {0,0,0,0};
//  unsigned int rm = initialAddr;
//  for (unsigned int i = 0; i < 4; i++) {
//    coord[i] = rm/dstPitch[i];
//    rm = rm-coord[i]*dstPitch[i];
//  }
//
//  unsigned int k = 4; //If it is a padding position we compute next useful
//  position for (unsigned int j = 3; j > 0; j--) {
//    if (coord[j] >= dstIndex[j]) {
//      coord[j-1]++;
//      k = j;
//    }
//  }
//  for (unsigned int j = k; j < 4; j++) coord[j] = 0;
//
//  bool done = false;
//  ssize_t dx, dy, x, y;
//  dx = kernel_minionId/kernels[1] - ssize_t(pads[0]);
//  dy = kernel_minionId%kernels[1] - ssize_t(pads[1]);
//  while(!done) {
//
//    x = coord[1]*strides[0] + dx;
//    y = coord[2]*strides[1] + dy;
//
//    auto sum = tAInput[0];
//    sum = 0;
//    if (x >= 0 && y >= 0 && x < ssize_t(actIndex[1]) &&
//        y < ssize_t(actIndex[2]) && kernel_minionId < numElemsKernel) {
//      sum = tAInput[coord[0]*actPitch[0] + x*actPitch[1] + y*actPitch[2] +
//      coord[3]*actPitch[3]];
//    }
//
//    for (int i = 0; i <= level; i++) {
//      sum = tensor_reduce_float(sum, 0x0, 1, i, 0x3);
//    }
//
//    if (kernel_minionId == 0) {
//      int64_t dstAddr = coord[0]*dstPitch[0] + coord[1]*dstPitch[1] +
//                         coord[2]*dstPitch[2] + coord[3]*dstPitch[3];
//      float tmp = sum;
//      tmp *= invFilter;
//      sum = tmp;
//      tOutput[dstAddr] = sum;
//    }
//
//    for (int j = 3; j >= 0; j--) {
//      if (coord[j] != (dstIndex[j] - 1)) {
//        offsetOut += dstPitch[j];
//        coord[j]++;
//        break;
//      } else if (j == 0) {
//        done = true;
//        break;
//      } else {
//        offsetOut -= (dstIndex[j] - 1) * dstPitch[j];
//        coord[j] = 0;
//      }
//    }
//    if (offsetOut >= posMax) break;
//
//  }
//}

template <typename srcType, typename dstType>
void dnn_lib::fwdLibAvgPoolInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *pkernels, void *pstrides, void *ppads, float *scale, int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<dstType> tOutput(dstMatrix, scale[1], offset[1]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides;
  unsigned int *pads = (unsigned int *)ppads;

  float filterArea = kernels[0] * kernels[1];
  float invFilter;
  fpReciprocalSingleElement(filterArea, invFilter);

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[4] = {0, 0, 0, 0};
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 4, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y;
  while (!done && (offsetOut < posMax)) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);

    auto sum = tAInput[0];
    sum = 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {
      for (size_t fy = 0; fy < kernels[1]; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }

        sum += tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                       (size_t)oy * actPitch[2] + coord[3] * actPitch[3]];
      }
    }

    float tmp = sum;
    tmp *= invFilter;
    sum = tmp;
    tOutput[offsetOut] = sum;

    done = getOffsets(4, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

//===----------------------------------------------------------------------===//
//                       Activation functions
//===----------------------------------------------------------------------===//
template <typename srcType>
void dnn_lib::fwdLibSigmoidInstThreaded(void *dstT, void *dstDims,
                                        void *dstPitches, void *srcT1,
                                        void *srcDims, void *srcPitches,
                                        unsigned int srcDimNum, float *scale,
                                        int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT1(srcT1, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  float op, inverse;
  while (!done && (offsetOut < posMax)) {
    op = getExp(-ptrSrcT1[offsetIn]) + 1.0;
    fpReciprocalSingleElement(op, inverse);
    ptrDstT[offsetOut] = inverse;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibSigmoidInst(void *dstT, void *dstDims, void *dstPitches,
                                void *srcT1, void *srcDims, void *srcPitches,
                                unsigned int srcDimNum, float *scale,
                                int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT1(srcT1, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;
  float op, inverse;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              op = getExp(-ptrSrcT1[addrSrc]) + 1.0;
              fpReciprocalSingleElement(op, inverse);
              ptrDstT[addrDst] = inverse;
            }
          }
        }
      }
    }
  }
}

// TODO Check corner cases
template <typename srcType>
void dnn_lib::fwdLibTanhInst(void *dstT, void *dstDims, void *dstPitches,
                             void *srcT1, void *srcDims, void *srcPitches,
                             unsigned int srcDimNum, float *scale,
                             int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> ptrSrcT1(srcT1, scale[0], offset[0]);
  Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  float op1, op2;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              op1 = getSinh(ptrSrcT1[x * eSrcPitch[0] + y * eSrcPitch[1] +
                                     z * eSrcPitch[2] + w * eSrcPitch[3] +
                                     q * eSrcPitch[4] + r * eSrcPitch[5]]);
              op2 = getCosh(ptrSrcT1[x * eSrcPitch[0] + y * eSrcPitch[1] +
                                     z * eSrcPitch[2] + w * eSrcPitch[3] +
                                     q * eSrcPitch[4] + r * eSrcPitch[5]]);
              fpReciprocalSingleElement(op2, op2);
              ptrDstT[x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                      w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5]] =
                  op1 * op2;
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibTanhInstThreaded(void *dstT, void *dstDims,
                                     void *dstPitches, void *srcT1,
                                     void *srcDims, void *srcPitches,
                                     unsigned int srcDimNum,
                                     float *scale, int32_t *offset,
                                     uint64_t flags) {


  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  float op1, op2;
  while (!done && (offsetOut < posMax)) {
    op1 = getSinh(aSrcT1[offsetIn]);
    op2 = getCosh(aSrcT1[offsetIn]);
    fpReciprocalSingleElement(op2, op2);
    ptrDstT[offsetOut] = op1 * op2;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

//===----------------------------------------------------------------------===//
//                        Loss Functions (Softmax/regression/...)
//===----------------------------------------------------------------------===//

template <typename srcType>
void dnn_lib::fwdLibSoftMaxInst(void *dstT, void *srcT, void *srcTDims,
                                void *srcTPitches, float *scale,
                                int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  const Addresser<srcType> acumInt(dstT, scale[1], offset[1]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);

  unsigned int *srcIndex = (unsigned int *)srcTDims;
  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  float e, sum, inverseSum;

  for (unsigned int n = 0; n < srcIndex[0]; n++) {
    unsigned int start = n * srcPitch[0];
    unsigned int end = start + srcIndex[1];

    // Find Max.
    float max = float(tInput[start]);
    for (unsigned int i = start + 1; i < end; i++)
      max = std::max(max, float(tInput[i]));

    // Compute exp.
    sum = 0;
    for (unsigned int i = 0; i < srcIndex[1]; i++) {
      e = getExp(float(tInput[n * srcPitch[0] + i]) - max);
      sum += e;
      tOutput[n * srcPitch[0] + i] = float(e);
    }

    fpReciprocalSingleElement(sum, inverseSum);
    // Normalize the output.
    for (unsigned int i = 0; i < srcIndex[1]; i++) {
      auto in = acumInt[n * srcPitch[0] + i];
      in = in * inverseSum;
      tOutput[n * srcPitch[0] + i] = in;
    }
  }
}

// Single-thread version with small optimisations. Useful when the padding
// hypothesis are not met.
template <typename srcType>
void dnn_lib::fwdLibSoftMaxInst2(void *dstT, void *srcT, void *srcTDims,
                                 void *srcTPitches, float *scale,
                                 int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  const Addresser<srcType> acumInt(dstT, scale[1], offset[1]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);

  unsigned int *srcIndex = (unsigned int *)srcTDims;
  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  float e, sum, inverseSum;

  for (unsigned int n = 0; n < srcIndex[0]; n++) {
    unsigned int start = n * srcPitch[0];
    unsigned int end = start + srcIndex[1];

    float max = float(tInput[start]);
    for (unsigned int i = start + 1; i < end; i++)
      max = std::max(max, float(tInput[i]));

    // Compute exp.
    sum = 0;
    for (unsigned int i = start; i < end; i++) {
      e = getExp(float(tInput[i]) - max);
      sum += e;
      tOutput[i] = float(e); // here, the shape hypothesis is important.
    }

    fpReciprocalSingleElement(sum, inverseSum);
    // Normalize the output.
    for (unsigned int i = start; i < end; i++) {
      auto in = acumInt[i];
      in = in * inverseSum;
      tOutput[i] = in;
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibSoftMaxInstThreaded(void *dstT, void *srcT, void *srcTDims,
                                        void *srcTPitches, float *scale,
                                        int32_t *offset, uint64_t flags) {
  
  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  size_t typeSize = getsize<srcType>();
  unsigned int cll = 64/typeSize;
  if (srcPitch[0]%cll == 0)  
    fwdLibSoftMaxInstThreaded1<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset, flags);
  else if (cll%srcPitch[0] == 0)
    fwdLibSoftMaxInstThreaded2<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset, flags);
  else fwdLibSoftMaxInst2<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset);
}

template <typename srcType>
void dnn_lib::fwdLibSoftMaxInstVectorized(void *dstT, void *srcT, void *srcTDims,
                                          void *srcTPitches, float *scale, 
                                          int32_t *offset, uint64_t flags) {
  
  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  size_t typeSize = getsize<srcType>();
  unsigned int cll = 64/typeSize;
  if (srcPitch[0]%cll == 0)  
    fwdLibSoftMaxInstVectorized1<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset, flags);
  else if (cll%srcPitch[0] == 0) // TODO: vectorize v2.
    fwdLibSoftMaxInstThreaded2<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset, flags);
  else fwdLibSoftMaxInst2<srcType>(dstT, srcT, srcTDims,
                                        srcTPitches, scale,
                                        offset);
}


  // Hypothesis 1 (SHAPE): both source and destination tensors have the same
  // padding pattern (and the same pitches, since they have the same
  // dimensions). Hypothesis 2 (COHERENCE): each row of the source tensor
  // contains an integer number of cl's, that is, srcPitch[0] is a multiple of
  // the cache line length. Thus, dividing the minions' work by rows guarantees
  // there will not be two minions in different rows writing on the same cache
  // line.

template <typename srcType>
void dnn_lib::fwdLibSoftMaxInstThreaded1(void *dstT, void *srcT, void *srcTDims,
                                         void *srcTPitches, float *scale,
                                         int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  const Addresser<srcType> acumInt(dstT, scale[1], offset[1]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  size_t typeSize = getsize<srcType>();

  unsigned int *srcIndex = (unsigned int *)srcTDims;
  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  float e, sum, inverseSum;

  unsigned int rowstodo = srcIndex[0] / activeMinions;
  unsigned int firstrow = minionId * rowstodo;
  unsigned int lastrow = firstrow + rowstodo;

  for (unsigned int n = firstrow; n < lastrow; n++) {
    unsigned int start = n * srcPitch[0];
    unsigned int end = start + srcIndex[1];

    float max = float(tInput[start]);
    for (unsigned int i = start + 1; i < end; i++)
      max = std::max(max, float(tInput[i]));

    // Compute exp.
    sum = 0;
    for (unsigned int i = start; i < end; i++) {
      e = getExp(float(tInput[i]) - max);
      sum += e;
      tOutput[i] = float(e); // here, the shape hypothesis is important.
    }

    fpReciprocalSingleElement(sum, inverseSum);
    // Normalize the output.
    for (unsigned int i = start; i < end; i++) {
      auto in = acumInt[i];
      in = in * inverseSum;
      tOutput[i] = in;
    }
  }

  unsigned int cll = 64 / getsize<srcType>();
  unsigned int clperpitch = srcPitch[0] / cll;
  if (DO_EVICTS) {
    unsigned int clperminion = clperpitch * rowstodo;
    unsigned int initialAddr = firstrow * srcPitch[0];
    evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
  }

  unsigned int doneRows = activeMinions * rowstodo;
  unsigned int remainingrows = srcIndex[0] - doneRows;

  if (remainingrows != 0) {
    //    unsigned int cll = 64/sizeof(srcType);
    unsigned int clperrow = (srcIndex[1] - 1) / cll + 1;

    unsigned int minionsperrow = 1;
    int level = -1; // level = log2(minionsperrow) - 1. This is a useful
                    // parameter for reducing and broadcasting.
    unsigned int aux =
        activeMinions /
        (2 * remainingrows); // This parameter helps guarantee that there are
                             // enough minions to double minionsperrow.
    while ((minionsperrow <= aux) && (minionsperrow <
               clperrow)) { // If possible (mpr < aux), we double mpr until we
                           // have at least 1 minion per cl.
      minionsperrow *= 2;
      ++level;
    }

    if (minionId >= minionsperrow * remainingrows)
      return;

    unsigned int moreRows = minionId / minionsperrow;
    unsigned int minioninrow = minionId - moreRows*minionsperrow;
    unsigned int type1minions = clperrow % minionsperrow;
    unsigned int clperminion;
    unsigned int K; // Number of skipped tensor elements by a minion in its own row.
    if (minioninrow < type1minions) {
      clperminion = ((clperrow - 1) / minionsperrow) + 1;
      K = minioninrow * clperminion * cll;
    } else {
      clperminion = clperrow / minionsperrow;
      K = (type1minions + minioninrow * clperminion) * cll;
    }

    // Starting and ending positions in the tensor which the minion will write on.
    unsigned int start = (doneRows + moreRows) * srcPitch[0] + K;
    unsigned int end = start + clperminion * cll;
    // If the minion is the last working minion in its row, its ending position
    // should be modified so it avoids padding.
    if ((clperrow < minionsperrow) && (minioninrow == clperrow - 1) ||
        (clperrow >= minionsperrow) && (minioninrow == minionsperrow - 1))
      end = start - K + srcIndex[1];

    // Now, we perform the SoftMax operation, using shared information between
    // minions.
    float max = float(
        tInput[start - K]); // Obseration: this way, if a minion has start =
                            // end (it is not assigned any cl's), the
                            // initialization of max will not affect the
                            // maximum value of its row (when reducing).
    for (unsigned int i = start; i < end; ++i)
      max = std::max(max, float(tInput[i]));

    // After reducing and broadcasting, the variable max will be the maximum
    // value in each the minion's row.
    for (int i = 0; i <= level; i++) {
      max = tensor_reduce_float(max, 0x2, 1, i, 0x3);
    }
    for (int i = level; i >= 0; i--) {
      max = tensor_reduce_float(max, 0x8, 1, i, 0x2);
    }

    sum = 0;
    for (unsigned int i = start; i < end; i++) {
      e = getExp(float(tInput[i]) - max);
      sum += e;
      tOutput[i] = float(e);
    }
    // Again, after reducing and broadcasting, the variable sum will be the
    // total sum of each the minion's row.
    for (int i = 0; i <= level; i++) {
      sum = tensor_reduce_float(sum, 0x0, 1, i, 0x3);
    }
    if (minionId % minionsperrow == 0)
      fpReciprocalSingleElement(
          sum, inverseSum); // only the first minion must do this calculation,
                            // for saving power and stores.
    for (int i = level; i >= 0; i--) {
      inverseSum = tensor_reduce_float(inverseSum, 0x8, 1, i, 0x2);
    }

    // Finally, the output is normalized.
    for (unsigned int i = start; i < end; i++) {
      auto in = acumInt[i];
      in = in * inverseSum;
      tOutput[i] = in;
    }
    if (clperminion > 0 && DO_EVICTS)
      evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*start, clperminion);
  }
}


template <typename srcType>
void dnn_lib::fwdLibSoftMaxInstThreaded2 (void *dstT, void *srcT, void *srcTDims,
                                          void *srcTPitches, float *scale,
                                          int32_t *offset, uint64_t flags) {
  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  const Addresser<srcType> acumInt(dstT, scale[1], offset[1]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);

  unsigned int *srcIndex = (unsigned int *)srcTDims;
  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  size_t typeSize = getsize<srcType>();
  unsigned int cll = 64/typeSize;
  unsigned int rowspercl = (cll - 1)/srcPitch[0] + 1;
  unsigned int rowstodo = rowspercl;
  while(activeMinions*rowstodo < srcIndex[0]) rowstodo += rowspercl;
  
  unsigned int firstrow = minionId*rowstodo;
  if (firstrow >= srcIndex[0]) 
    return;
  unsigned int lastrow = firstrow + rowstodo;  
  if (lastrow > srcIndex[0])
    lastrow = srcIndex[0];

  float e, sum, inverseSum;
  
  for (unsigned int n = firstrow; n < lastrow; n++) {
    unsigned int start = n * srcPitch[0];
    unsigned int end = start + srcIndex[1];

    float max = float(tInput[start]);
    for (unsigned int i = start + 1; i < end; i++)
      max = std::max(max, float(tInput[i]));

    // Compute exp.
    sum = 0;
    for (unsigned int i = start; i < end; i++) {
      e = getExp(float(tInput[i]) - max);
      sum += e;
      tOutput[i] = float(e); // here, the shape hypothesis is important.
    }

    fpReciprocalSingleElement(sum, inverseSum);
    // Normalize the output.
    for (unsigned int i = start; i < end; i++) {
      auto in = acumInt[i];
      in = in * inverseSum;
      tOutput[i] = in;
    }
  }
}

// Vectorized version: same hypothesis than SoftMaxInstThreaded1.
// Possible source types: fp16, fp32 (and output of the same type).
// TODO: use templates for each srcType for a speed up (the only
// difference is in the GATHER_FLOAT and SCATTER_FLOAT functions).

template <typename srcType>
void dnn_lib::fwdLibSoftMaxInstVectorized1(void *dstT, void *srcT, void *srcTDims,
                                           void *srcTPitches, float *scale,
                                           int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  const Addresser<srcType> acumInt(dstT, scale[1], offset[1]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);

  unsigned int *srcIndex = (unsigned int *)srcTDims;
  unsigned int *srcPitch = (unsigned int *)srcTPitches;

  size_t typeSize = getsize<srcType>();
  float e, sum, inverseSum;

  unsigned int rowstodo = srcIndex[0] / activeMinions;
  unsigned int firstrow = minionId * rowstodo;
  unsigned int lastrow = firstrow + rowstodo;

  unsigned int step = srcPitch[0]*typeSize;
  unsigned int memOffset = firstrow*step;
  uintptr_t srcAddr = (uintptr_t)srcT + memOffset;
  uintptr_t dstAddr = (uintptr_t)dstT + memOffset;

  unsigned int numRegs = srcIndex[1]/8;
  unsigned int extraLanes = srcIndex[1] - 8*numRegs;
  bool floatType = (typeSize == 4); // 1 if fp32 and 0 if fp16.
  int32_t registerSize = 8*typeSize;
  float log2e = M_LOG2E;

#define GATHER_FLOAT(_addr)                       \
      "beq %[floatType], zero, 16f \n"            \
      "flw.ps f0, 0x0(" #_addr ") \n"             \
      "j 32f \n"                                  \
      "16: \n"                                    \
      "fg32h.ps f0, t0(" #_addr ") \n"            \
      "fcvt.ps.f16 f0, f0 \n"                     \
      "32: \n"

#define SCATTER_FLOAT                             \
      "beq %[floatType], zero, 16f \n"            \
      "fsw.ps f0, 0x0(%[dstAddr]) \n"             \
      "j 32f \n"                                  \
      "16: \n"                                    \
      "fcvt.f16.ps f0, f0 \n"                     \
      "fsc32h.ps f0, t0(%[dstAddr]) \n"           \
      "32: \n"

#define EXP                                       \
      "fadd.ps f0, f0, f29 \n"                    \
      "fmul.ps f0, f0, f27 \n"                    \
      "fexp.ps f0, f0 \n"

#define DO_REG(_op, _reg)                         \
      "mov.m.x m0, zero, 0xff \n"                 \
      "fswizz.ps    f27, " #_reg ", 0xe \n"       \
      "f" #_op ".ps " #_reg ", f27, " #_reg " \n" \
      "fswizz.ps    f27, " #_reg ", 0x1 \n"       \
      "f" #_op ".ps " #_reg ", f27, " #_reg " \n" \
      "fmvs.x.ps    t1, " #_reg ", 0x4 \n"        \
      "fmv.w.x      f27, t1 \n"                   \
      "f" #_op ".s  " #_reg ", f27, " #_reg " \n" \
      "fmvs.x.ps    t1, " #_reg ", 0x0 \n"        \
      "fbcx.ps      " #_reg ", t1 \n"

  for (unsigned int n = firstrow; n < lastrow; n++) {

    __asm__ __volatile__(
      "mov.m.x m0, zero, 0xff \n"
      "fxor.pi f28, f28, f28 \n"
      SET_MINUS_INFTY(f29)
      
///// PART 1: COMPUTATION OF THE MAX VALUE IN ROW.
      "addi t1, zero, 0x0 \n"
      "add t2, zero, %[srcAddr] \n"
      "add t3, zero, %[dstAddr] \n"

      "addi t0, zero, 0x1 \n"
      "beq %[floatType], t0, 1f \n"
      SET_FG32H_VAL(t0)

      "1: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 2f \n"
      GATHER_FLOAT(%[srcAddr])
      "fmax.ps f29, f0, f29 \n"
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 1b \n"

      "2: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 3f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[srcAddr])
      "fmax.ps f29, f0, f29 \n"

      "3: \n" // Computation of the max value.
      DO_REG(max, f29)
      "fsub.ps f29, f28, f29 \n" // f29 = -max.

///// PART 2: COMPUTATION OF EXPONENTIALS AND ITS SUM.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddr], zero, t2 \n"
      "add %[dstAddr], zero, t3 \n"

      "fbc.ps f27, 0x0(%[log2e]) \n"

      "4: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 5f \n"
      GATHER_FLOAT(%[srcAddr])
      EXP
      "fadd.ps f28, f0, f28 \n"
      SCATTER_FLOAT
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 4b \n"

      "5: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 6f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[srcAddr])
      EXP
      "fadd.ps f28, f0, f28 \n"
      SCATTER_FLOAT

      "6: \n" // Computation of the sum of exponentials.
      DO_REG(add, f28)
      "frcp.ps f28, f28 \n" // Reciprocal of the sum of exps.

///// PART 3: PRODUCT OF THE EXPONENTIALS BY THE INVERSE SUM.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddr], zero, t2 \n"
      "add %[dstAddr], zero, t3 \n"

      "7: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 8f \n"
      GATHER_FLOAT(%[dstAddr])
      "fmul.ps f0, f0, f28 \n"
      SCATTER_FLOAT
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 7b \n"

      "8: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 9f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[dstAddr])
      "fmul.ps f0, f0, f28 \n"
      SCATTER_FLOAT

      "9: \n"

      :
      : [log2e] "r" (&log2e),
        [registerSize] "r" (registerSize),
        [floatType] "r" (floatType),
        [srcAddr] "r" (srcAddr),
        [dstAddr] "r" (dstAddr),
        [numRegs] "r" (numRegs),
        [extraLanes] "r" (extraLanes)
      : "t0", "t1", "t2", "t3", "f0", "f27", "f28", "f29", "memory");
  
    srcAddr += step;
    dstAddr += step;
  }

  unsigned int doneRows = activeMinions * rowstodo;
  unsigned int remainingrows = srcIndex[0] - doneRows;

  if (remainingrows != 0) {

    unsigned int cll = 64 / getsize<srcType>();
    unsigned int clperrow = (srcIndex[1] - 1) / cll + 1;

    unsigned int minionsperrow = 1;
    int level = -1;
    unsigned int aux = activeMinions / (2 * remainingrows);

 // If possible (mpr < aux), we double mpr until we have at least 1 minion per cl.
    while ((minionsperrow <= aux) && (minionsperrow < clperrow)) {
      minionsperrow *= 2;
      ++level;
    }

    if (minionId >= minionsperrow * remainingrows)
      return;

    unsigned int moreRows = minionId / minionsperrow;
    unsigned int minioninrow = minionId - moreRows*minionsperrow;
    unsigned int type1minions = clperrow % minionsperrow;
    unsigned int clperminion;
    unsigned int K; // Number of skipped tensor elements by a minion in its own row.
    if (minioninrow < type1minions) {
      clperminion = ((clperrow - 1) / minionsperrow) + 1;
      K = minioninrow * clperminion * cll;
    } else {
      clperminion = clperrow / minionsperrow;
      K = (type1minions + minioninrow * clperminion) * cll;
    }

    // Starting and ending positions in the tensor which the minion will write on.
    unsigned int start = (doneRows + moreRows) * srcPitch[0] + K;
    unsigned int end = start + clperminion * cll;
    // If the minion is the last working minion in its row, its ending position
    // should be modified so it avoids padding.
    if ((clperrow < minionsperrow) && (minioninrow == clperrow - 1) ||
        (clperrow >= minionsperrow) && (minioninrow == minionsperrow - 1))
      end = start - K + srcIndex[1];

    memOffset = start*typeSize;
    srcAddr = (uintptr_t)srcT + memOffset;
    dstAddr = (uintptr_t)dstT + memOffset;
    numRegs = (end - start)/8;
    extraLanes = (end - start) - 8*numRegs;

    uint64_t csr_enc = ((0ULL  & 0x2) << 62)        |
                       ((29ULL & 0x1F) << 57)       |   // Register: f29.
                       ((0ULL  & 0x1FFFFFFF) << 28) |
                       ((2ULL  & 0xF) << 24)        |   // Op: 0 = add, 2 = max
                       ((1ULL  & 0xFF) << 16)       |   // Number of registers
                       ((0ULL  & 0x1FFF) << 3)      |   // Tree depth
                       ((0ULL  & 0x1) << 2)         |
                       ((0x3 & 0x3));                   // 2: broadcast, 3:reduce

#define REDUCE                                    \
      "addi t4, zero, 0x0 \n"                     \
      "1: \n"                                     \
      "blt %[level], t4, 2f \n"                   \
      "csrw tensor_reduce, t1 \n"                 \
      "addi t1, t1, 0x8 \n"                       \
      "addi t4, t4, 0x1 \n"                       \
      "j 1b \n"                                   \
      "2: \n"                                     \
      "addi t1, t1, -8 \n"                        \
      "addi t4, t4, -1 \n"

#define BROADCAST                                 \
      "1: \n"                                     \
      "blt t4, zero, 2f \n"                       \
      "csrw tensor_reduce, t1 \n"                 \
      "addi t1, t1, -8 \n"                        \
      "addi t4, t4, -1 \n"                        \
      "j 1b \n"                                   \
      "2: \n"
    
    __asm__ __volatile__(
      "mov.m.x m0, zero, 0xff \n"
      "fxor.pi f28, f28, f28 \n"
      SET_MINUS_INFTY(f29)

///// PART 1: COMPUTATION OF THE MAX VALUE IN ROW.
      "addi t1, zero, 0x0 \n"
      "add t2, zero, %[srcAddr] \n"
      "add t3, zero, %[dstAddr] \n"

      "addi t0, zero, 0x1 \n"
      "beq %[floatType], t0, 1f \n"
      SET_FG32H_VAL(t0)

      "1: \n" // Coverage of the full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 2f \n"
      GATHER_FLOAT(%[srcAddr])
      "fmax.ps f29, f0, f29 \n"
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 1b \n"

      "2: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 3f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[srcAddr])
      "fmax.ps f29, f0, f29 \n"

      "3: \n" // Computation of the max value.
      DO_REG(max, f29)

      "ld t1, 0x0(%[csr_enc]) \n"
      REDUCE
      // Changing reduce max to broadcast get.
      "addi t1, t1, -1 \n"
      "addi t5, zero, 6 \n"
      "slli t5, t5, 0x18 \n"
      "add t1, t1, t5 \n"
      BROADCAST

      "fsub.ps f29, f28, f29 \n" // f29 = -max.

///// PART 2: COMPUTATION OF EXPONENTIALS AND ITS SUM.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddr], zero, t2 \n"
      "add %[dstAddr], zero, t3 \n"

      "fbc.ps f27, 0x0(%[log2e]) \n"

      "4: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 5f \n"
      GATHER_FLOAT(%[srcAddr])
      EXP
      "fadd.ps f28, f0, f28 \n"
      SCATTER_FLOAT
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 4b \n"

      "5: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 6f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[srcAddr])
      EXP
      "fadd.ps f28, f0, f28 \n"
      SCATTER_FLOAT

      "6: \n" // Computation of the sum of exponentials.
      DO_REG(add, f28)

      "ld t1, 0x0(%[csr_enc]) \n"
      // Changing f29 to f28.
      "addi t4, zero, 1 \n"
      "slli t4, t4, 0x39 \n"
      "sub t1, t1, t4 \n"
      // Changing max to add.
      "addi t4, zero, 2 \n"
      "slli t4, t4, 0x18 \n"
      "sub t1, t1, t4 \n"
      REDUCE
      // Changing reduce add to broadcast get.
      "addi t1, t1, -1 \n"
      "addi t5, zero, 8 \n"
      "slli t5, t5, 0x18 \n"
      "add t1, t1, t5 \n"
      BROADCAST

      "frcp.ps f28, f28 \n" // Reciprocal of the sum of exps.

///// PART 3: PRODUCT OF THE EXPONENTIALS BY THE INVERSE SUM.
      "addi t1, zero, 0x0 \n"
      "add %[srcAddr], zero, t2 \n"
      "add %[dstAddr], zero, t3 \n"

      "7: \n" // Coverage of the srcIndex[1]/8 full registers.
      "addi t1, t1, 0x1 \n"
      "blt %[numRegs], t1, 8f \n"
      GATHER_FLOAT(%[dstAddr])
      "fmul.ps f0, f0, f28 \n"
      SCATTER_FLOAT
      "add %[srcAddr], %[srcAddr], %[registerSize] \n"
      "add %[dstAddr], %[dstAddr], %[registerSize] \n"
      "j 7b \n"

      "8: \n" // Coverage of the last srcIndex[1]%8 extra lanes.
      "beq %[extraLanes], zero, 9f \n"
      "addi t1, zero, 0x1 \n"
      "sll t1, t1, %[extraLanes] \n"
      "addi t1, t1, -1 \n"
      "mov.m.x m0, t1, 0 \n"
      GATHER_FLOAT(%[dstAddr])
      "fmul.ps f0, f0, f28 \n"
      SCATTER_FLOAT

      "9: \n"

      :
      : [log2e] "r" (&log2e),
        [registerSize] "r" (registerSize),
        [floatType] "r" (floatType),
        [srcAddr] "r" (srcAddr),
        [dstAddr] "r" (dstAddr),
        [numRegs] "r" (numRegs),
        [extraLanes] "r" (extraLanes),
        [csr_enc] "r" (&csr_enc),
        [level] "r" (level)
      : "t0", "t1", "t2", "t3", "t4", "t5", "f0", "f27", "f28", "f29", "memory");

  }
#undef GATHER_FLOAT
#undef SCATTER_FLOAT
#undef EXP 
#undef DO_REG
#undef REDUCE
#undef BROADCAST
}

template <typename srcType>
void dnn_lib::fwdLibCrossEntropyLossInst(void *dstT, void *srcT, void *srcDims,
                                         void *srcPitches,
                                         unsigned int srcDimNum, void *labelsT,
                                         float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);
  const Addresser<srcType> tTmp(dstT, scale[2], offset[2]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  long long *tLabels = (long long *)labelsT;

  unsigned int *srcIndex = (unsigned int *)srcDims;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  float op1;
  const float op2 = M_1_LOG2E;

  for (size_t n = 0; n < srcIndex[0]; ++n) {
    size_t y = tLabels[n];
    float p_n = float(tInput[n * srcPitch[0] + y]);
    fpLog2SingleElement(p_n, op1);
    float mulOp = op1 * op2;
    auto tmp = tTmp[0];
    tmp -= mulOp;
    tOutput[0] = tmp;
  }
}

template <typename srcType>
void dnn_lib::fwdLibCrossEntropyLossInstThreaded(
    void *dstT, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, void *labelsT, float *scale, int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);
  const Addresser<srcType> tTmp(dstT, scale[2], offset[2]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  long long *tLabels = (long long *)labelsT;

  unsigned int *srcIndex = (unsigned int *)srcDims;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int rowstodo = srcIndex[0] / activeMinions;
  unsigned int firstrow;
  unsigned int type1minions = srcIndex[0] - rowstodo * activeMinions;
  if (minionId < type1minions) {
    ++rowstodo;
    firstrow = minionId * rowstodo;
  } else
    firstrow = type1minions +
               minionId * rowstodo; // Simplification of type1minions*(rowstodo
                                    // + 1) + (minionId - type1minions)*rowstodo
  unsigned int lastrow = firstrow + rowstodo;

  float op;
  float sum = 0;
  unsigned int rowaddress =
      firstrow * srcPitch[0]; // address of the first element of the row
                              // considered in the following loop.
  for (size_t n = firstrow; n < lastrow; ++n) {
    float p_n = float(tInput[rowaddress + tLabels[n]]);
    fpLog2SingleElement(p_n, op);
    sum -= op;
    rowaddress += srcPitch[0];
  }

  unsigned int level = 0;
  for (int k = 1; k < activeMinions; k *= 2)
    level++;

  for (int i = 0; i < level; i++)
    sum = tensor_reduce_float(sum, 0x0, 1, i, 0x3);
  if (minionId == 0)
    tOutput[0] = sum * M_1_LOG2E;
}

//===----------------------------------------------------------------------===//
//                                 Checksum
//===----------------------------------------------------------------------===//

static inline void uint32_to_ascii_hex(char *s, uint32_t value) {
  for (uint32_t i = 0; i < 8; i++) {
    uint32_t bits = (value >> (28 - i * 4)) & 0xF;
    s[i] = (bits <= 9) ? ('0' + bits) : ('A' + bits - 10);
  }
}

template <typename srcType>
void dnn_lib::fwdLibChecksum(void *src, void *srcDims, void *srcPitches,
                             unsigned int srcDimNum, float *scale,
                             int32_t *offset, uint64_t flags) {
  // The checksum is the u32 addition of all the non-padding bytes of the tensor
  uint32_t checksum = 0;

  unsigned int *actIndex = (unsigned int *)srcDims;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  size_t typeSize = getsize<srcType>();
  unsigned int srcNumElems = actIndex[0] * actPitch[0];

  unsigned int minionElems = srcNumElems / activeMinions;
  unsigned int mod = srcNumElems - minionElems * activeMinions;
  unsigned int startPos;
  if (minionId < mod) {
    ++minionElems;
    startPos = minionElems * minionId;
  } else
    startPos = mod + minionId * minionElems;

  if (minionElems != 0) {
    unsigned int endPos = startPos + minionElems;

    unsigned int coordIn[srcDimNum];
    unsigned int lastNonZeroCoord;
    getNonPaddingCoordinates(coordIn, startPos, srcDimNum, actPitch, actIndex,
                             lastNonZeroCoord);

    unsigned int offsetIn = 0;
    for (unsigned int i = 0; i < lastNonZeroCoord; i++)
      offsetIn += actPitch[i] * coordIn[i];

    bool done = false;
    while (!done && (offsetIn < endPos)) {
      uint8_t *elemAddr = (uint8_t *)src + offsetIn * typeSize;
      // Iterate over all the bytes of the element
      for (size_t i = 0; i < typeSize; i++) {
        checksum += (uint32_t)*(elemAddr + i);
      }
      done = getOffsets(srcDimNum, coordIn, offsetIn, actIndex, actPitch);
    }
  }

  // Reduce CheckSum across active minions
  // TODO: make this general for non power of two active shires
  int level = 4;
  for (int i = 1; i < ACTIVE_SHIRES; i *= 2)
    ++level;
  for (int i = 0; i <= level; i++)
    checksum = tensor_reduce_uint32(checksum, TENSOR_REDUCE_OP_IADD, i, 0x3);

  // Convert Checksum to ASCII and dump via UART1
  if (minionId == 0) {
    char chs_str[] = "CheckSum: 0xXXXXXXXX\n";
    uint32_to_ascii_hex(&chs_str[12], checksum);
    ecall(FW_SCODE_ECALL_LOG_WRITE, (uint64_t)chs_str, sizeof(chs_str), 0);
  }
}

// Node to force flushing the L3 at the end of the computation
void dnn_lib::fwdLibFlushL3(uint32_t numShires) {
  uint32_t minion = get_minion_id() & 0x1F;
  // The T0 of minion N of shire 0 flushes the L3 of shire N
  if ((get_shire_id() == 0) && (get_thread_id() == 0) && (minion < numShires)) {
      ecall(FW_COMPUTE_SCODE_ECALL_L3_FLUSH_ALL, minion, 0, 0);
  }
}



//===----------------------------------------------------------------------===//
//                       Tensor shape (copy/transpose/concat/...)
//===----------------------------------------------------------------------===//

template <typename srcType>
void dnn_lib::fwdLibCopyInst(void *dst, void *src, unsigned int numElems,
                             float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tInput(src, scale[0], offset[0]);

  for (int i = 0; i < numElems; i += 1) {
    tOutput[i] = tInput[i];
  }
}


/* This is currently the fastest version of the copy, geting up to 800x velocity
 * compared to the single thread version of CopyInst. It splits the total
 * cachelines in packs and distributes them between all the minions possible.
 *
 * It is specially more desirable than other threading implementations
 * due to the fact that it updates the position being read and written via
 * the sum of the pitch, instead of computing it on each iteration of the loop.
 *
 * Moreover, this version gets over the limited convention of only considering
 * arrays of 6 dimensions, as extended vectors are not needed, and therefore
 * this implementation is a generalization of the previous ones. */

template <typename srcType>
void dnn_lib::fwdLibCopyInstThreaded(void *dst, void *dstDims,
                                     void *dstPitches, void *src,
                                     void *srcDims, void *srcPitches,
                                     unsigned int srcDimNum, float *scale,
                                     int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  uint8_t *dst8 = (uint8_t *)dst;
  uint8_t *src8 = (uint8_t *)src;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst =
      dstPitch[0] * dstIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address and the number of positions that
  // it must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position

  unsigned int k;                  // Amount of non-zero coordinates
  unsigned int coord[srcDimNum]; // Vector of coordinates

  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    tOutput[offsetOut] = tAInput[offsetIn];
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


// Vectorized and threaded general version of CopyInst. Work in progress.

template <typename srcType>
void dnn_lib::fwdLibCopyInstVectorized(void *dst, void *dstDims,
                                       void *dstPitches, void *src,
                                       void *srcDims, void *srcPitches,
                                       unsigned int srcDimNum, float *scale,
                                       int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;
  
  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;
  uint8_t *dst8 = (uint8_t *)dst;
  uint8_t *src8 = (uint8_t *)src;
  uint8_t *src8Init = src8;
  uint8_t *dst8Init = dst8;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int typeSize = getsize<srcType>();
  unsigned int numElemsDst =
      dstPitch[0] * actIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;
  // We move the initialAddr to the next non-padding position
  unsigned int k;                  // Amount of non-zero coordinates
  unsigned int coord[srcDimNum]; // Vector of coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;

  unsigned int laneElems = 4 / typeSize;
  unsigned int registerElems;
  if (laneElems != 0) {
    registerElems = 8 * laneElems;
  } else {
    registerElems = 4;
  }
  unsigned int maxRow = (srcDimNum > 1) ? posMax / dstPitch[lastDim - 1] : 0;
  unsigned int elementsInRow, registersInRow, res, spareElems, fullLanes;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false; 
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

  while ((offsetOut < posMax) && !done) {
    if (lastDim != 0) {
      if (firstRow && (coord[lastDim - 1] != maxRow)) {
        elementsInRow = dstIndex[lastDim] - coord[lastDim];
      } else if (coord[lastDim - 1] == maxRow) {
        lastRow = true;
        elementsInRow = posMax - offsetOut;
      } 
      else {
      elementsInRow = dstIndex[lastDim];
      }
    } else {
      elementsInRow = dstIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / registerElems;
      res = elementsInRow - registersInRow * registerElems;
      if (laneElems != 0) {
        fullLanes = res / laneElems;
        spareElems = res - fullLanes * laneElems;
      } else {
        fullLanes = res * 2;
        spareElems = 0;
      }
      mask = ((1 << fullLanes) - 1);
      __asm__ __volatile__("mov.m.x m1, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    src8 += offsetIn * typeSize; 
    dst8 += offsetOut * typeSize;

    unsigned int cnt = 0;
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
    while (cnt < registersInRow) {
      __asm__ __volatile__("flw.ps f0, 0x0(%[src]) \n"
                           "fsw.ps f0, 0x0(%[dst]) \n"
                           :
                           : [ src ] "r"(src8), [ dst ] "r"(dst8)
                           : "f0", "memory");
      src8 += 32;
      dst8 += 32;
      cnt++;
    }
    __asm__ __volatile__("maskand m0, m0, m1 \n"
                         "flw.ps f0, 0x0(%[src]) \n"
                         "fsw.ps f0, 0x0(%[dst]) \n"
                         "mov.m.x m0, zero, 0xff \n"
                         :
                         : [ src ] "r"(src8), [ dst ] "r"(dst8)
                         : "f0", "memory");
    src8 += fullLanes * 4;
    dst8 += fullLanes * 4;
    unsigned int offsetInAux = (src8 - src8Init) / typeSize;
    unsigned int offsetOutAux = (dst8 - dst8Init) / typeSize;
    for (unsigned int i = 0; i < spareElems; i++) {
      tOutput[offsetOutAux + i] = tAInput[offsetInAux + i];
    }

    if (lastRow) 
      return;
    src8 = src8Init;
    dst8 = dst8Init;
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;

    done = getOffsets(srcDimNum - 1, coord, offsetIn, offsetOut, actIndex, actPitch, dstPitch);
  }
}

// template <typename srcType>
// void dnn_lib::fwdLibCopyInstVectorizedGeneral(void *dst, void *dstDims,
//                                        void *dstPitches, void *src,
//                                        void *srcDims, void *srcPitches,
//                                        unsigned int srcDimNum, float *scale,
//                                        int32_t *offset, uint64_t flags) {
//   Addresser<srcType> tOutput(dst, scale[1], offset[1]);
//   const Addresser<srcType> tAInput(src, scale[0], offset[0]);
// 
//   unsigned int *dstIndex = (unsigned int *)dstDims;
//   unsigned int *actIndex = (unsigned int *)srcDims;
//   int8_t *dst8 = (int8_t *)dst;
//   int8_t *src8 = (int8_t *)src;
// 
//   unsigned int *dstPitch = (unsigned int *)dstPitches;
//   unsigned int *actPitch = (unsigned int *)srcPitches;
// 
//   unsigned int minionId = get_minion_id();
//   unsigned int activeMinions = 32 * ACTIVE_SHIRES;
//   if (minionId >= activeMinions) {
//     return;
//   }
//   unsigned int typeSize = getsize<srcType>();
//   unsigned int numElemsDst =
//       dstPitch[0] * actIndex[0]; // Total number of elements in the tensor
// 
//   // We give to each minion an initial address the number of positions that it
//   // must work on (maxRead).
//   unsigned int initialAddr, maxRead;
//   getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
//                         activeMinions);
//   if (maxRead == 0)
//     return;
//   // We move the initialAddr to the next non-padding position
//   unsigned int k;                  // Amount of non-zero coordinates
//   unsigned int coord[srcDimNum]; // Vector of coordinates
//   getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
//                            k);
// 
//   // We get the actual initialAddr, in the input and output.
//   unsigned int offsetIn = 0;
//   unsigned int offsetOut = 0;
//   for (unsigned int j = 0; j < k; j++) {
//     offsetIn += actPitch[j] * coord[j];
//     offsetOut += dstPitch[j] * coord[j];
//   }
// 
//   unsigned int posMax = maxRead + initialAddr;
//   bool done = false;
//   unsigned int lastDim = srcDimNum - 1;
//   if (actIndex[lastDim] < 4) {
//     while ((offsetOut < posMax) && (not done)) {
//       tOutput[offsetOut] = tAInput[offsetIn];
//       done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
//                         actPitch, dstPitch);
//     }
//   } else {
//     while ((coord[lastDim] != 0) && (offsetOut < posMax) && (not done)) {
//       tOutput[offsetOut] = tAInput[offsetIn];
//       done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
//                         actPitch, dstPitch);
//     }
//     unsigned int registerElems = 32 / typeSize;
//     unsigned int res = dstIndex[lastDim] % registerElems;
//     unsigned int maxAux = posMax - res;
//     unsigned int limit = dstIndex[lastDim] - res;
//     src8 += offsetIn * typeSize;
//     dst8 += offsetOut * typeSize;
//     __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
//     __asm__ __volatile__("mov.m.x m1, zero, 0xff \n");
//     uint8_t mask = (uint8_t)((1 << res) - 1);
//     __asm__ __volatile__("mov.m.x m2, %[mask], 0 \n" : : [ mask ] "r"(&mask) :);
//     while ((offsetOut < maxAux) && (not done)) {
//       if (coord[lastDim] < limit) {
//         __asm__ __volatile__("flw.ps f0, 0x0(%[src]) \n"
//                              "fsw.ps f0, 0x0(%[dst]) \n"
//                              :
//                              : [ src ] "r"(src8), [ dst ] "r"(dst8)
//                              : "f0");
//         src8 += 32;
//         dst8 += 32;
//         coord[lastDim]++; // += registerElems
//       } else {
//         __asm__ __volatile__("maskand m0, m0, m2 \n"
//                              "flw.ps f0, 0x0(%[src]) \n"
//                              "fsw.ps f0, 0x0(%[dst]) \n"
//                              "maskor m0, m0, m1 \n"
//                              :
//                              : [ src ] "r"(src8), [ dst ] "r"(dst8)
//                              : "f0");
//         unsigned int i = srcDimNum - 2;
//         src8 -= coord[lastDim] * typeSize;
//         dst8 -= coord[lastDim] * typeSize;
//         coord[lastDim] = 0;
//         if ((coord[i] + 1) != dstIndex[i]) {
//           coord[i]++;
//           offsetIn += actPitch[i];
//           offsetOut += dstPitch[i];
//           src8 += actPitch[i] * typeSize;
//           dst8 += dstPitch[i] * typeSize;
//         } else {
//           while ((coord[i] + 1) == dstIndex[i]) {
//             if (i != 0) {
//               coord[i] = 0;
//               offsetOut -= (dstIndex[i] - 1) * dstPitch[i];
//               offsetIn -= (actIndex[i] - 1) * actPitch[i];
//               src8 -= (actIndex[i] - 1) * typeSize * actPitch[i];
//               dst8 -= (dstIndex[i] - 1) * typeSize * dstPitch[i];
//               i--;
//               coord[i]++;
//               offsetOut += dstPitch[i];
//               offsetIn += actPitch[i];
//               src8 += actPitch[i] * typeSize;
//               dst8 += dstPitch[i] * typeSize;
//             } else {
//               done = true;
//               break;
//             }
//           }
//         }
//       }
//     }
//     if (! done) {
//       offsetOut += coord[lastDim]; // Due to addresser, last pitch = 1
//       while (not done && offsetOut < posMax) {
//         tOutput[offsetOut] = tAInput[offsetIn];
//         done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
//                           actPitch, dstPitch);
//       }
//     }
//   }
// }
     
 // This implementation takes advantage of small cases with the same input and
 // output shape. It does not try to avoid padding, as the calculations needed
 // would decrease velocity. Therefore, we just give each minion its initial
 // address and make it copy everything until maxRead. In the end, there should
 // be a graph decision between this version and the general vectorisation.
 
 template <typename srcType>
 void dnn_lib::fwdLibCopyInstTensorized(void *dst, void *dstDims, void *dstPitches,
                                        void *src, void *srcDims, void *srcPitches,
                                        unsigned int srcDimNum, float *scale,
                                        int32_t *offset, uint64_t flags) {
  
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;
 
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;
 
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;
 
  size_t typeSize = getsize<srcType>();
  uint64_t numElemsDst = dstPitch[0] * actIndex[0] *
                             typeSize; // Total number of elements in the tensor
  uint64_t numCacheLines = (numElemsDst - 1) / 64 + 1; //64 = CacheLineLength
  uint64_t minionCacheLines = (numCacheLines - 1) / activeMinions + 1;
  uint64_t initialCacheLine = minionCacheLines * minionId;
  uint64_t lastCacheLine = initialCacheLine + minionCacheLines;
  minionCacheLines = 
          (lastCacheLine <= numCacheLines) ? minionCacheLines 
        : (initialCacheLine < numCacheLines) ? numCacheLines - initialCacheLine : 0;
  uint64_t srcAddr = (uint64_t)src + initialCacheLine*64;
  uint64_t dstAddr = (uint64_t)dst + initialCacheLine*64;

  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

  while (minionCacheLines >= 16) {
    tensor_load(0, 0, 0, 0, 0, srcAddr, 0, 0xF, 0x40, 0);
    tensor_store_scp(0, 0, 0xF, dstAddr, 0x40);
    srcAddr += 1024;
    dstAddr += 1024;
    minionCacheLines -= 16;
  }
  if (minionCacheLines == 0) return;
  tensor_load(0, 0, 0, 0, 0, srcAddr, 0, minionCacheLines-1, 0x40, 0);
  tensor_store_scp(0, 0, minionCacheLines-1, dstAddr, 0x40);
}
 

template <typename srcType>
void dnn_lib::fwdLibTransposeInst(void *dst, void *dstDims, void *dstPitches,
                                  void *src, void *srcDims, void *srcPitches,
                                  unsigned int srcDimNum, void *pshuffle,
                                  float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int *shuffle = (unsigned int *)pshuffle;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  // Iterates through all dimensions, and sets extended Dims, and src Pitches
  for (int i = 0; i < srcDimNum; i++) {
    // extended Dims matches src dim and zeros for non used dims
    eDims[i] = actIndex[i];
    // extended src Pitches matches src Pitches and zeros for non used dims
    eSrcPitch[i] = actPitch[i];
    for (int j = 0; j < srcDimNum; j++) {
      if (shuffle[j] == i) {
        eDstPitch[i] = dstPitch[j];
      }
    }
  }

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              uint64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                 z * eDstPitch[2] + w * eDstPitch[3] +
                                 q * eDstPitch[4] + r * eDstPitch[5];
              uint64_t srcAddr = x * eSrcPitch[0] + y * eSrcPitch[1] +
                                 z * eSrcPitch[2] + w * eSrcPitch[3] +
                                 q * eSrcPitch[4] + r * eSrcPitch[5];
              tOutput[dstAddr] = tAInput[srcAddr];
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibTransposeInstThreaded(void *dst, void *dstDims,
                                          void *dstPitches, void *src,
                                          void *srcDims, void *srcPitches,
                                          unsigned int srcDimNum,
                                          void *pshuffle, float *scale,
                                          int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int *shuffle = (unsigned int *)pshuffle;

  uintptr_t dstAddr = (uintptr_t)dst;
  uintptr_t srcAddr = (uintptr_t)src;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int newPitch[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    newPitch[i] = actPitch[shuffle[i]];

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex,
                           k);

  
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += newPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (not done && offsetOut < posMax) {
    tOutput[offsetOut] = tAInput[offsetIn];
    done = getOffsets(srcDimNum, coord, offsetOut, offsetIn, dstIndex, dstPitch, newPitch);       
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}




template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value, 
std::size_t>::type = 0>
void transposeOp (uintptr_t dst, uintptr_t src, volatile int32_t *scatterValues, volatile int32_t *gatherValues){
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues]) \n"    
                       "fgw.ps  f0, f31(%[src]) \n"             
                       "flw.ps f31, 0x0(%[scatterValues]) \n"
                       "fscw.ps  f0, f31(%[dst]) \n"
                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ scatterValues ] "r"(scatterValues),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "f0", "f31", "memory");
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, 
std::size_t>::type = 0>
void transposeOp (uintptr_t dst, uintptr_t src, volatile int32_t *scatterValues, volatile int32_t *gatherValues){
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues]) \n"    
                       "fgh.ps  f0, f31(%[src]) \n"             
                       "flw.ps f31, 0x0(%[scatterValues]) \n"
                       "fsch.ps  f0, f31(%[dst]) \n"
                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ scatterValues ] "r"(scatterValues),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "f0", "f31", "memory");
  
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value, 
std::size_t>::type = 0>
void transposeOp (uintptr_t dst, uintptr_t src, volatile int32_t *scatterValues, volatile int32_t *gatherValues){
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues]) \n"    
                       "fgb.ps  f0, f31(%[src]) \n"             
                       "flw.ps f31, 0x0(%[scatterValues]) \n"
                       "fscb.ps  f0, f31(%[dst]) \n"
                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ scatterValues ] "r"(scatterValues),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "f0", "f31", "memory");
  
}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value 
&& !std::is_same<srcType, float16>::value 
&& !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void transposeOp (uintptr_t dst, uintptr_t src, volatile int32_t *scatterValues, volatile int32_t *gatherValues){}



template <typename srcType>
void dnn_lib::fwdLibTransposeInstVectorized(void *dst, void *dstDims,
                                            void *dstPitches, void *src,
                                            void *srcDims, void *srcPitches,
                                            unsigned int srcDimNum,
                                            void *pshuffle, float *scale,
                                            int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  uintptr_t dstAddr = (uintptr_t)dst;
  uintptr_t srcAddr = (uintptr_t)src;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int *shuffle = (unsigned int *)pshuffle;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int newPitch[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    newPitch[i] = actPitch[shuffle[i]];
  
  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += newPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;
  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;

  
  unsigned int elementsInRow, registersInRow, res;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false; 
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

 
  unsigned int newPitchSize = newPitch[lastDim] * typeSize;
  volatile int32_t gatherValues[8];
  for (unsigned int i = 0; i < 8; i++) gatherValues[i] = i*newPitchSize;
  unsigned int dstPitchSize = dstPitch[lastDim] * typeSize;
  volatile int32_t scatterValues[8];
  for (unsigned int i = 0; i < 8; i++) scatterValues[i] = i*dstPitchSize;

  while (!done && (offsetOut < posMax)) {
    if (firstRow && (coord[lastDim - 1] != maxRow)) {
      elementsInRow = dstIndex[lastDim] - coord[lastDim];
    } else if (coord[lastDim - 1] == maxRow) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
    } else {
      elementsInRow = dstIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      res = elementsInRow - registersInRow * 8;
      mask = ((1 << res) - 1);
      __asm__ __volatile__("mov.m.x m1, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr += offsetIn * typeSize; 
    dstAddr += offsetOut * typeSize;

    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

    for (unsigned int i = 0; i < registersInRow; i++) {
      transposeOp <srcType>(dstAddr, srcAddr, scatterValues, gatherValues);
      srcAddr += 8 * typeSize * newPitch[lastDim];
      dstAddr += 8 * typeSize;
    }

    if (res > 0) {
      __asm__ __volatile__("maskand m0, m1, m0 \n");
      transposeOp <srcType>(dstAddr, srcAddr, scatterValues, gatherValues);
    }

    if (lastRow) 
      return;
    
    dstAddr = (uintptr_t)dst;
    srcAddr = (uintptr_t)src;
    offsetIn -= coord[lastDim] * newPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;

    done = getOffsets(lastDim, coord, offsetOut, offsetIn, dstIndex, dstPitch, newPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value, 
std::size_t>::type = 0>
void transposeOpAligned32Bytes (uintptr_t dst, uintptr_t src, volatile int32_t *gatherValues){
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues]) \n"    
                       "fgw.ps  f0, f31(%[src]) \n"             
                       "fsw.ps  f0, 0x0(%[dst]) \n"
                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "f0", "f31", "memory");
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, 
std::size_t>::type = 0>
void transposeOpAligned32Bytes (uintptr_t dst, uintptr_t src, volatile int32_t *gatherValues){
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues]) \n"    
                       "fgh.ps  f0, f31(%[src]) \n" 
                       SET_FG32H_VAL(t0)
                       "fsc32h.ps f0, t0(%[dst]) \n"
                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "t0", "f0", "f31", "memory");
	
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value, 
std::size_t>::type = 0>
void transposeOpAligned32Bytes (uintptr_t dst, uintptr_t src, volatile int32_t *gatherValues){
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues]) \n"    
                       "fgb.ps  f0, f31(%[src]) \n"
                       SET_FG32B_VAL(t0)             
                       "fsc32b.ps  f0, t0(%[dst]) \n"
                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "t0", "f0", "f31", "memory");
	
}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value 
&& !std::is_same<srcType, float16>::value 
&& !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void transposeOpAligned32Bytes (uintptr_t dst, uintptr_t src, volatile int32_t *gatherValues){}



template <typename srcType>
void dnn_lib::fwdLibTransposeInstAligned32Bytes(void *dst, 
                                          void *dstDims,
                                          void *dstPitches, void *src,
                                          void *srcDims, void *srcPitches,
                                          unsigned int srcDimNum,
                                          void *pshuffle, float *scale,
                                          int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  uintptr_t dstAddr = (uintptr_t)dst;
  uintptr_t srcAddr = (uintptr_t)src;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int *shuffle = (unsigned int *)pshuffle;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int newPitch[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    newPitch[i] = actPitch[shuffle[i]];
  
  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += newPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  unsigned int lastDim = srcDimNum - 1;
  unsigned int newPitchSize = newPitch[lastDim] * typeSize;
  volatile int32_t gatherValues[8];
  for (unsigned int i = 0; i < 8; i++) gatherValues[i] = i*newPitchSize;

  //We modify the pitches and coord so that the function getOffsets 
  //jumps eight positions in lastDim, the smallest dimension.
  //Number 8 is the amount of lanes that a register has. 
  unsigned int res = ((dstIndex[lastDim] - 1)%8) + 1;
  newPitch[lastDim] *= 8;
  dstPitch[lastDim] *= 8;
  dstIndex[lastDim] = (dstIndex[lastDim] - 1)/8 + 1;
  unsigned int mask = ((1 << res) - 1);
  
  while (!done && (offsetOut < posMax)) {
    dstAddr = (uintptr_t)dst + offsetOut*typeSize;
    srcAddr = (uintptr_t)src + offsetIn*typeSize;

    //When the minion reaches the end of the lastDim, we use a mask 
    //that is always the same because the dst Tensor is aligned to 32 Bytes.
    if (coord[lastDim] != dstIndex[lastDim] - 1) 
         __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
    else __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);

    transposeOpAligned32Bytes <srcType>(dstAddr, srcAddr, gatherValues); 
    done = getOffsets(srcDimNum, coord, offsetOut, offsetIn, dstIndex, dstPitch, newPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0)
    evict_va(0, DO_EVICTS, initialAddr, clperminion - 1, 64); 
}



template <typename srcType>
void dnn_lib::fwdLibTensorViewInst(void *dst, void *dstDims, void *dstPitches,
                                   unsigned int dstDimNum, void *src,
                                   void *srcDims, void *srcPitches,
                                   unsigned int srcDimNum, void *pcoord,
                                   float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  long unsigned int *coord = (long unsigned int *)pcoord;

  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);
  int offsetIn = 0;

  unsigned int eDstDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcCnt[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (int i = 0; i < MAX_TENSOR_DIMENSIONS; i++) {
    if (i < dstDimNum) {
      eDstDims[i] = dstIndex[i];
      eDstPitch[i] = dstPitch[i];
    }
    if (i < srcDimNum) {
      offsetIn += coord[i] * actPitch[i];
      eSrcCnt[i] = coord[i];
    }
  }
  bool done = false;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eDstDims[0]; x++) {
    for (size_t y = 0; y < eDstDims[1]; y++) {
      for (size_t z = 0; z < eDstDims[2]; z++) {
        for (size_t w = 0; w < eDstDims[3]; w++) {
          for (size_t q = 0; q < eDstDims[4]; q++) {
            for (size_t r = 0; r < eDstDims[5]; r++) {
              if (!done) {
                uint64_t addr = x * eDstPitch[0] + y * eDstPitch[1] +
                                z * eDstPitch[2] + w * eDstPitch[3] +
                                q * eDstPitch[4] + r * eDstPitch[5];
                tOutput[addr] = tAInput[offsetIn];
                for (int j = srcDimNum - 1; j >= 0; j--) {
                  if (eSrcCnt[j] != (actIndex[j] - 1)) {
                    offsetIn += actPitch[j];
                    eSrcCnt[j]++;
                    break;
                  } else if (j == 0) {
                    done = true;
                    break;
                  } else {
                    offsetIn -= (actIndex[j] - 1) * actPitch[j];
                    eSrcCnt[j] = 0;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibTensorViewInstThreaded(
    void *dst, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src, void *srcDims, void *srcPitches, unsigned int srcDimNum,
    void *pcoord, float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  long unsigned int *coord = (long unsigned int *)pcoord;
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddrOut, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddrOut, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coordIn[srcDimNum];
  unsigned int coordOut[dstDimNum];
  unsigned int kOut = 0;
  getNonPaddingCoordinates(coordOut, initialAddrOut, dstDimNum, dstPitch,
                           dstIndex, kOut);

  unsigned int auxActPitch[srcDimNum];           //Pitch not accounting padding
  auxActPitch[srcDimNum - 1] = 1;
  for (int i = srcDimNum - 2; i >= 0; i--)
    auxActPitch[i] = auxActPitch[i + 1] * actIndex[i + 1];

  unsigned int auxdstPitch[dstDimNum];           //Pitch not accounting padding
  auxdstPitch[dstDimNum - 1] = 1;
  for (int i = dstDimNum - 2; i >= 0; i--)
    auxdstPitch[i] = auxdstPitch[i + 1] * dstIndex[i + 1];

  unsigned int addrIn = 0;
  unsigned int addrOut = 0;
  unsigned int elements_moved = 0;
  for (unsigned int j = 0; j < kOut; j++) {
    addrOut += dstPitch[j] * coordOut[j];
    elements_moved += auxdstPitch[j] * coordOut[j];
  }

  for (unsigned int i = 0; i < srcDimNum; i++) {
    coordIn[i] = elements_moved / auxActPitch[i];
    elements_moved = elements_moved - coordIn[i] * auxActPitch[i];
  }
  for (int i = srcDimNum - 1; i >= 0; i--) {
    coordIn[i] += (int)coord[i];
    if (coordIn[i] >= actIndex[i]) {
      coordIn[i] = coordIn[i] % actIndex[i];
      coordIn[i - 1] += 1;
    }
    addrIn += coordIn[i] * actPitch[i];
  }
  unsigned int posMax = maxRead + initialAddrOut;

  bool done = false;
  bool donein = false;
  while (!done && (addrOut < posMax)) {
    tOutput[addrOut] = tAInput[addrIn];
    donein = getOffsets(srcDimNum, coordIn, addrIn, actIndex, actPitch);
    done = getOffsets(dstDimNum, coordOut, addrOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddrOut, clperminion);
}

inline __attribute__((always_inline)) 
  unsigned int minTview(uint8_t &type, unsigned int a, unsigned int b, 
                        unsigned int c) {
  type = 0;
  if(b < a) {
    type = 1;
    a = b;
  }
  if(c < a) {
    type = 2;
    return c;
  }
  return a;
}

template <typename srcType>
inline __attribute__((always_inline)) void 
gatherScatterTView(uint8_t *src8, uint8_t *dst8, const uint32_t &mask, 
                   int32_t *gatherValues) {
  if (getsize<srcType>() == 2) {
    __asm__ __volatile__(
        "mov.m.x m0, %[mask], 0\n"
        "flw.ps f1, 0x0(%[gatherValues]) \n"    
        "fgh.ps f0, f1(%[src]) \n"              
        "fsch.ps f0, f1(%[dst]) \n"             
        :                                       
        : [ src ] "r"(src8), [ dst ] "r"(dst8), 
          [ gatherValues ] "r"(gatherValues),
          [ mask ] "r" (mask)    
        : "f0", "f1", "memory"); 
  } else if (getsize<srcType>() == 1) {
    __asm__ __volatile__(
        "mov.m.x m0, %[mask], 0\n"
        "flw.ps f1, 0x0(%[gatherValues]) \n"    
        "fgb.ps f0, f1(%[src]) \n"              
        "fscb.ps f0, f1(%[dst]) \n"             
        :                                       
        : [ src ] "r"(src8), [ dst ] "r"(dst8), 
          [ gatherValues ] "r"(gatherValues),
          [ mask ] "r" (mask)    
        : "f0", "f1", "memory");
  }
  return;
}

template <typename srcType>
inline __attribute__((always_inline)) void
getLanesResTView (int &lanes, int &res, const unsigned int &d) {
  if (getsize<srcType>() == 1) {
    lanes = d / 4;
    res = d - 4 * lanes;
  } else if (getsize<srcType>() == 2) {
    lanes = d / 2;
    res = d - 2 * lanes;
  } else if (getsize<srcType>() == 4) {
    lanes = d;
    res = 0;
  } else if (getsize<srcType>() == 8) {
    lanes = d * 2;
    res = 0;
  }
  return;
}


template <typename srcType>
void dnn_lib::fwdLibTensorViewInstVectorized(
    void *dst, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src, void *srcDims, void *srcPitches, unsigned int srcDimNum,
    void *pcoord, float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return; // Minion not working

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  long unsigned int *coord = (long unsigned int *)pcoord;
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddrOut, maxRead;
  int32_t typeSize = (int32_t) getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddrOut, maxRead,
                        minionId, activeMinions); // Obtain initial addr 4 minions
  if (maxRead == 0)
    return; // No work to do for the minion

  unsigned int coordIn[srcDimNum], coordOut[dstDimNum], kOut;
  getNonPaddingCoordinates(coordOut, initialAddrOut, dstDimNum, dstPitch,
                           dstIndex, kOut); // Find the first useful coord vect
  unsigned int srcLastDim = srcDimNum - 1;
  unsigned int dstLastDim = dstDimNum - 1;
  unsigned int auxActPitch[srcDimNum], auxdstPitch[dstDimNum];
  auxActPitch[srcLastDim] = auxdstPitch[dstLastDim] = 1;
  for (int i = srcDimNum - 2; i >= 0; i--)
    auxActPitch[i] = auxActPitch[i + 1] * actIndex[i + 1];
  for (int i = dstDimNum - 2; i >= 0; i--)
    auxdstPitch[i] = auxdstPitch[i + 1] * dstIndex[i + 1];

  unsigned int addrIn, addrOut, elements_moved;
  addrIn = addrOut = elements_moved = 0;

   for (unsigned int j = 0; j < kOut; j++) { // Compute the output address
    addrOut += dstPitch[j] * coordOut[j]; 
    elements_moved += auxdstPitch[j] * coordOut[j];
  }
  for (unsigned int i = 0; i < srcDimNum; i++) { // Compute the input coord vec
    coordIn[i] = elements_moved / auxActPitch[i]; 
    elements_moved = elements_moved - coordIn[i] * auxActPitch[i];
  }
  for (int i = srcLastDim; i >= 0; i--) { // Add to in coord vect the offset
    coordIn[i] += (int)coord[i]; 
    if (coordIn[i] >= actIndex[i]) {
      coordIn[i] = coordIn[i] % actIndex[i];
      coordIn[i - 1] += 1;
    }
    addrIn += coordIn[i] * actPitch[i]; // Compute input address
  }
  uint8_t *dst8 = (uint8_t *) dst + addrOut*typeSize;;
  uint8_t *src8 = (uint8_t *) src + addrIn*typeSize;
  unsigned int posMax = std::min(maxRead + initialAddrOut, numElemsDst); // Last position to "copy"
  maxRead = posMax - addrOut;
  
  int32_t gatherValues[8] = { 0, typeSize, 2 * typeSize, 3 * typeSize, 
                               4 * typeSize, 5 * typeSize, 6 * typeSize,
                               7 * typeSize}; //Computed at compilation time

  bool done = false;
  while ((addrOut < posMax) & !done) {
    uint8_t type;
    unsigned int d = minTview(type, maxRead, 
                               actIndex[srcLastDim] - coordIn[srcLastDim],
                               dstIndex[dstLastDim] - coordOut[dstLastDim]);
    if(type == 1) {
      addrOut += (d - 1) * dstPitch[dstLastDim];
      coordOut[dstLastDim] += (d - 1);
      addrIn -= coordIn[srcLastDim] * actPitch[srcLastDim];
      coordIn[srcLastDim] = 0;
    }
    else if (type == 2) {
      addrIn += (d - 1) * actPitch[srcLastDim];
      coordIn[srcLastDim] += (d - 1);
      addrOut -= coordOut[dstLastDim] * dstPitch[dstLastDim];
      coordOut[dstLastDim] = 0;
    }

    maxRead -= d; // FIXME it does not support doubles
    int lanes, res;
    getLanesResTView <srcType> (lanes, res, d);
    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n");
    while (lanes >= 8) {
      __asm__ __volatile__("flw.ps f0, 0x0(%[src])\n"
                           "fsw.ps f0, 0x0(%[dst])\n"
                           :
                           : [ src ] "r"(src8), [ dst ] "r"(dst8)
                           : "f0", "memory");
      lanes -= 8;
      src8 += 32;
      dst8 += 32;
    }
    if (lanes != 0) {                        
      uint32_t mask = ((1 << lanes) - 1); 
      __asm__ __volatile__(
        "mov.m.x m0, %[mask], 0\n"
        "flw.ps f0, 0x0(%[src])\n"              
        "fsw.ps f0, 0x0(%[dst])\n"              
        :                                       
        : [ src ] "r"(src8), [ dst ] "r"(dst8),
          [ mask ] "r" (mask)  
        : "f0", "memory");
      src8 += 4*lanes;
      dst8 += 4*lanes;
    }
    if (res != 0) {                        
      uint8_t mask = ((1 << res) - 1);
      gatherScatterTView <srcType>(src8, dst8, mask, gatherValues); 
    }
    if (type == 0) 
      break;
    if (type == 1) {
      done = getOffsets(srcLastDim, coordIn, addrIn, actIndex, actPitch);
      done = getOffsets(dstDimNum, coordOut, addrOut, dstIndex, dstPitch);
    }
    else if(type == 2) {
      done = getOffsets(srcDimNum, coordIn, addrIn, actIndex, actPitch);
      done = getOffsets(dstLastDim, coordOut, addrOut, dstIndex, dstPitch);
      // TODO this last getOffsets could be avoided, as in lines 4144, 4145
      // we could use d instead of d - 1. This yields no problems as in
      // the min function type == 2 iff the inequality is strict
    }
    src8 = (uint8_t *) src + typeSize * addrIn;
    dst8 = (uint8_t *) dst + typeSize * addrOut;
  }

  if (!DO_EVICTS) // Evicting the result
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddrOut, clperminion);
}


template <typename srcType>
void dnn_lib::fwdLibSplatInst(void *addr, int numElems, float splatVal,
                              float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(addr, scale[0], offset[0]);
  for (int i = 0; i < numElems; i++) {
    tOutput[i] = splatVal;
  }
}

template <typename srcType>
void dnn_lib::fwdLibSplatInst(void *addr, int numElems, int64_t splatVal,
                              float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  srcType *tOutput = (srcType *)addr;
  for (int i = 0; i < numElems; i++) {
    tOutput[i] = splatVal;
  }
}

template <typename srcType>
void dnn_lib::fwdLibSplatInstThreaded(void *dst, void *dstDims,
                                      void *dstPitches, unsigned int dstDimNum,
                                      float splatVal, float *scale,
                                      int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

 Addresser<srcType> tOutput(dst, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  // Get minion id
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[dstDimNum]; // Vector of coordinates
  unsigned int k;                  // Amount of non-zero coordinates

  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    tOutput[offsetOut] = splatVal;
    done = getOffsets(dstDimNum, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibSplatInstThreaded(void *dst, void *dstDims,
                                      void *dstPitches, unsigned int dstDimNum,
                                      int64_t splatVal, float *scale,
                                      int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  srcType *tOutput = (srcType *)dst;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  // Get minion id
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;
  unsigned int coord[dstDimNum]; // Vector of coordinates
  unsigned int k;                  // Amount of non-zero coordinates

  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    tOutput[offsetOut] = splatVal;
    done = getOffsets(dstDimNum, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibSplatInstVectorized(void *dst, void *dstDims,
                                        void *dstPitches, unsigned int dstDimNum,
                                        float splatVal, float *scale,
                                        int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  size_t typeSize = getsize<srcType>();
  size_t bytesperCL = 64;

  uint64_t totalBytes = dstPitch[0] * dstIndex[0] * typeSize;
  uint64_t totalCL = (totalBytes - 1)/bytesperCL + 1;
  uint64_t CLperMinion = (totalCL - 1)/activeMinions + 1;
  uint64_t startCL = minionId * CLperMinion;

  if (startCL >= totalCL) return;
  if (startCL + CLperMinion > totalCL) CLperMinion = totalCL - startCL;

  uint64_t regsperMinion = 2 * CLperMinion;  // A cacheline contains 2 regs
  uint64_t offsetOut = startCL * 64;
  uint64_t startElem = offsetOut/typeSize;
  char *dstPtr = (char *)dst;
  dstPtr += offsetOut;

  size_t numVals = 8/typeSize;
  for (unsigned int j = 0; j < numVals; j++)
    tOutput[startElem + j] = splatVal;
    
  __asm__ __volatile__("mov.m.x m0, zero, 0x55\n"
                       "fbc.ps f0, 0x0(%[dstPtr])\n"
                       "mov.m.x m0, zero, 0xaa\n"
                       "fbc.ps f0, 0x4(%[dstPtr])\n"
                       "mov.m.x m0, zero, 0xff\n"

                       "beq %[regs], zero, 1f\n"
                       "2:\n"
                       "fsw.ps f0, 0x0(%[dstPtr])\n"
                       "addi %[dstPtr], %[dstPtr], 0x20\n"
                       "addi %[regs], %[regs], -1\n"
                       "bne %[regs], zero, 2b\n"
                       "1:\n"
           
                       : [dstPtr] "+r"(dstPtr),
                         [regs] "+r"(regsperMinion)
                       :
                       : "f0");
}

template <typename srcType>
void dnn_lib::fwdLibSplatInstVectorized(void *dst, void *dstDims,
                                        void *dstPitches, unsigned int dstDimNum,
                                        int64_t splatVal, float *scale,
                                        int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();

  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  srcType *tOutput = (srcType *)dst;
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  size_t typeSize = getsize<srcType>();
  size_t bytesperCL = 64;

  uint64_t totalBytes = dstPitch[0] * dstIndex[0] * typeSize;
  uint64_t totalCL = (totalBytes - 1)/bytesperCL + 1;
  uint64_t CLperMinion = (totalCL - 1)/activeMinions + 1;
  uint64_t startCL = minionId * CLperMinion;

  if (startCL >= totalCL) return;
  if (startCL + CLperMinion > totalCL) CLperMinion = totalCL - startCL;

  uint64_t regsperMinion = 2 * CLperMinion;  // A cacheline contains 2 regs
  uint64_t offsetOut = startCL * 64;
  uint64_t startElem = offsetOut/typeSize;
  char *dstPtr = (char *)dst;
  dstPtr += offsetOut;

  size_t numVals = 8/typeSize;
  for (unsigned int j = 0; j < numVals; j++)
    tOutput[startElem + j] = splatVal;
    
  __asm__ __volatile__("mov.m.x m0, zero, 0x55\n"
                       "fbc.ps f0, 0x0(%[dstPtr])\n"
                       "mov.m.x m0, zero, 0xaa\n"
                       "fbc.ps f0, 0x4(%[dstPtr])\n"
                       "mov.m.x m0, zero, 0xff\n"

                       "beq %[regs], zero, 1f\n"
                       "2:\n"
                       "fsw.ps f0, 0x0(%[dstPtr])\n"
                       "addi %[dstPtr], %[dstPtr], 0x20\n"
                                   "addi %[regs], %[regs], -1\n"
                       "bne %[regs], zero, 2b\n"
                       "1:\n"
                       
                       : [dstPtr] "+r"(dstPtr),
                         [regs] "+r"(regsperMinion)
                       :
                       : "f0");
}

template <typename srcType>
void dnn_lib::fwdLibInsertTensorInst(void *dst, void *dstDims, void *dstPitches,
                                     unsigned int dstDimNum, void *src2,
                                     void *src2Dims, void *src2Pitches,
                                     void *pcoord, unsigned int count,
                                     unsigned int axis, float *scale,
                                     int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tSmallInput(src2, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *smallIndex = (unsigned int *)src2Dims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *smallPitch = (unsigned int *)src2Pitches;

  unsigned int *coord = (unsigned int *)pcoord;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eOffsets[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < dstDimNum; i++) {
    eDims[i] = smallIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = smallPitch[i];
    eOffsets[i] = coord[i];
  }

  size_t advanceOnAxis = 0;

  for (size_t cnt = 0; cnt < count; cnt++) {
    // We can use this loop for all shapes.
    for (size_t x = 0; x < eDims[0]; x++) {
      for (size_t y = 0; y < eDims[1]; y++) {
        for (size_t z = 0; z < eDims[2]; z++) {
          for (size_t w = 0; w < eDims[3]; w++) {
            for (size_t q = 0; q < eDims[4]; q++) {
              for (size_t r = 0; r < eDims[5]; r++) {
                tOutput[(eOffsets[0] + x) * eDstPitch[0] +
                        (eOffsets[1] + y) * eDstPitch[1] +
                        (eOffsets[2] + z) * eDstPitch[2] +
                        (eOffsets[3] + w) * eDstPitch[3] +
                        (eOffsets[4] + q) * eDstPitch[4] +
                        (eOffsets[5] + r) * eDstPitch[5] + advanceOnAxis] =
                    tSmallInput[x * eSrcPitch[0] + y * eSrcPitch[1] +
                                z * eSrcPitch[2] + w * eSrcPitch[3] +
                                q * eSrcPitch[4] + r * eSrcPitch[5]];
              }
            }
          }
        }
      }
    }
    advanceOnAxis += eDstPitch[axis] * eDims[axis];
  }
}

//FIXME This version fits the small cases that currently are not vectorized,
//but it still fails some tests.
//template <typename srcType>
//void dnn_lib::fwdLibInsertTensorInstThreaded(void *dst, void *dstDims,
//                                             void *dstPitches,
//                                             unsigned int dstDimNum, void *src2,
//                                             void *src2Dims, void *src2Pitches,
//                                             void *poffsets, unsigned int count,
//                                             unsigned int axis, float *scale,
//                                             int32_t *offset, uint64_t flags) {
//  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
//  const Addresser<srcType> tAInput(src2, scale[0], offset[0]);
//
//  unsigned int *dstIndex = (unsigned int *)dstDims;
//  unsigned int *actIndex = (unsigned int *)src2Dims;
//
//  unsigned int *actPitch = (unsigned int *)src2Pitches;
//  unsigned int *dstPitch = (unsigned int *)dstPitches;
//  unsigned int *coord = (unsigned int *)poffsets;
//  
//  unsigned int minionId = get_minion_id();
//  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
//  if (minionId >= activeMinions)
//    return;
//  size_t typeSize = getsize<srcType>();
//  unsigned int cll = 64/typeSize;
//
//  //Computing initial and last Address for each minion
//  unsigned int helper = actIndex[axis];
//  actIndex[axis] *= count;
//  unsigned int lastPos, addrOffset;
//  if(axis == dstDimNum - 1) {
//    //TODO these two for should be merged in one. 
//    //FIXME It should depend on axis == n-1, where the lastPos should be in the
//    //next last element and in the axis != n-1, where the last element is as 
//    //here
//    addrOffset = lastPos =  0;
//    for (unsigned int i = 0; i < dstDimNum; i++) {
//      addrOffset += coord[i] * dstPitch[i];
//    }
//    // Is it really necessary the last dimension term?
//    for (unsigned int i = dstDimNum - 2; i < dstDimNum - 1; i++) { 
//      lastPos += (coord[i] + actIndex[i]) * 
//               dstPitch[i];
//    }
//    for (unsigned int i = 0; i < dstDimNum - 1; i++) {
//      lastPos += (coord[i] + actIndex[i] - 1) * dstPitch[i];
//    }
//  }
//  else {
//    lastPos = (coord[axis] + actIndex[axis]) * dstPitch[axis];
//    addrOffset = coord[axis] * dstPitch[axis];
//    for (unsigned int i = 0; i < axis; i++) {
//      lastPos += (coord[i] + actIndex[i] - 1) * dstPitch[i];
//      addrOffset += coord[i] * dstPitch[i];
//    }
//  }
//  
//  unsigned int moved = addrOffset % cll;
//  unsigned int ncl = (moved + lastPos - addrOffset - 1) / cll + 1;
//  unsigned int mcl = (ncl - 1) / activeMinions + 1;
//  unsigned int div = ncl / mcl;
//  unsigned int maxRead;
//  if (minionId < div) 
//    maxRead = mcl * cll;
//  else if (minionId == div) 
//    maxRead = (ncl - div * mcl) * cll;
//  else
//    return; 
//  unsigned int addrOut = addrOffset + maxRead * minionId;
//  unsigned int posMax = std::min(addrOut + maxRead - moved, lastPos);
//  if (minionId != 0){
//    addrOut -= moved;
//  }
//
//  //Jumping to the next useful position
//  unsigned int coordIn[dstDimNum], k, addrIn;
//  getNonPaddingCoordinates(coordIn, addrOut - addrOffset, dstDimNum, dstPitch, actIndex, k);
//  addrIn = addrOut = 0;
//  for (unsigned int i = 0; i < axis; i++) {
//    addrOut += (coord[i] + coordIn[i]) * dstPitch[i];
//    addrIn += coordIn[i] * actPitch[i];
//  }
//  addrOut += (coord[axis] + coordIn[axis]) * dstPitch[axis];
//  addrIn += (coordIn[axis] % helper) * actPitch[axis];
//  for (unsigned int i = axis + 1; i < dstDimNum; i++) {
//    addrIn += coordIn[i] * actPitch[i];
//    addrOut += (coord[i] + coordIn[i]) * dstPitch[i];
//  }
//
//  bool done = false;
//  while ((addrOut < posMax) && !done) {
//    tOutput[addrOut] = tAInput[addrIn];
//  // TODO try using two getoffsets functions in order to verify this is correct
//    for (int j = dstDimNum - 1; j >= 0; j--) {
//      if (coordIn[j] != (actIndex[j] - 1)) {
//        addrOut += dstPitch[j];
//        coordIn[j]++;
//        //if ((j != axis) || (coordIn[axis] % helper != 0))
//        addrIn += actPitch[j];
//        //TODO avoid this if and module every iteration with a counter
//        if ((j == axis) && (coordIn[axis] % helper == 0))
//          addrIn -= helper * actPitch[axis];
//        break;
//      } else if (j != 0) {
//        if (j != axis)
//          addrIn -= (actIndex[j] - 1) * actPitch[j];
//        else 
//          addrIn -= (helper - 1) * actPitch[axis]; 
//        addrOut -= (actIndex[j] - 1) * dstPitch[j];
//        coordIn[j] = 0;
//      } else 
//        done = true;
//    }
//  }
//}

template <typename srcType>
inline void insertRow(uint8_t *dst, uint8_t *src, const unsigned int& addrOut,
                      const unsigned int& addrIn, const int32_t& typeSize, 
                      int lanes, int res, int32_t *gatherValues) {
  uint8_t *dst8 = (uint8_t *) dst + addrOut * typeSize;
  uint8_t *src8 = (uint8_t *) src + addrIn * typeSize;
  __asm__ __volatile__("mov.m.x m0, zero, 0xff");
  while (lanes > 8) {
    __asm__ __volatile__("flw.ps f0, 0x0(%[src])\n"
                         "fsw.ps f0, 0x0(%[dst])\n"
                         :
                         : [ src ] "r"(src8), [ dst ] "r"(dst8)
                         : "f0", "memory");
    lanes -= 8;
    src8 += 32;
    dst8 += 32;
  }
  __asm__ __volatile__(
  "maskand m0, m1, m1\n"
  "flw.ps f0, 0x0(%[src])\n"              
  "fsw.ps f0, 0x0(%[dst])\n"              
  :                                       
  : [ src ] "r"(src8), [ dst ] "r"(dst8)  
  : "f0", "memory");
  src8 += 4*lanes;
  dst8 += 4*lanes;
  if (res != 0) {
    if (getsize<srcType>() == 2) {
      __asm__ __volatile__(
          "maskand m0, m2, m2\n"
          "flw.ps f1, 0x0(%[gatherValues]) \n"    
          "fgh.ps f0, f1(%[src]) \n"              
          "fsch.ps f0, f1(%[dst]) \n"             
          :                                       
          : [ src ] "r"(src8), [ dst ] "r"(dst8), 
            [ gatherValues ] "r"(gatherValues)
          : "f0", "f1", "memory"); 
    } else if (getsize<srcType>() == 1) {
      __asm__ __volatile__(
          "maskand m0, m2, m2\n"
          "flw.ps f1, 0x0(%[gatherValues]) \n"    
          "fgb.ps f0, f1(%[src]) \n"              
          "fscb.ps f0, f1(%[dst]) \n"             
          :                                       
          : [ src ] "r"(src8), [ dst ] "r"(dst8), 
            [ gatherValues ] "r"(gatherValues)
          : "f0", "f1", "memory");
    }
  }                       
}

template <typename srcType>
void dnn_lib::fwdLibInsertTensorInstThreaded(void *dst, void *dstDims,
                                             void *dstPitches,
                                             unsigned int dstDimNum, void *src2,
                                             void *src2Dims, void *src2Pitches,
                                             void *poffsets, unsigned int count,
                                             unsigned int axis, float *scale,
                                             int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;
  
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  int32_t typeSize = (int32_t) getsize<srcType>();
  unsigned int cll = 64/typeSize;

  if ((dstDimNum >= 2) && (dstPitch[dstDimNum - 2]%cll != 0)) {    
    fwdLibInsertTensorInst<srcType>(dst, dstDims, dstPitches,
                                     dstDimNum, src2,
                                     src2Dims, src2Pitches,
                                     poffsets, count,
                                     axis, scale,
                                     offset);
    return;   
  }

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src2, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)src2Dims;

  unsigned int *actPitch = (unsigned int *)src2Pitches;
  unsigned int *coord = (unsigned int *)poffsets;

  // We compute the offset address
  unsigned int offsetNum = coord[0] * dstPitch[0];
  for (unsigned int i = 1; i < dstDimNum; i++)
    offsetNum += coord[i] * dstPitch[i]; // Offset Address
  unsigned int jump = dstPitch[axis] * actIndex[axis]; 

  unsigned int dimRow = 0;
  if (dstDimNum > 1) 
    dimRow = dstDimNum - 2;
  unsigned int lastDim = dstDimNum - 1;

  int32_t gatherValues[8] = { 0, typeSize, 2 * typeSize, 3 * typeSize, 
                              4 * typeSize, 5 * typeSize, 6 * typeSize,
                              7 * typeSize};

  int lanes, res;
  getLanesResTView<srcType>(lanes, res, actIndex[lastDim]);
    
  uint32_t mask = (1 << (((lanes - 1) % 8) + 1)) - 1;
  __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                        :
                        : [ mask ] "r" (mask));
  mask = (1 << res) - 1;
  __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                        :
                        : [ mask ] "r" (mask));

  if (axis != lastDim) {
    unsigned int auxNRows = count * actIndex[0];
    for (int i = 1; i < lastDim; i++)
      auxNRows *= actIndex[i];
    unsigned int mRows = auxNRows / activeMinions; 
    unsigned int mod = auxNRows - activeMinions * mRows;
    if (minionId < mod) {
      ++mRows;
      mod = 0;
    }
    if (unlikely(mRows == 0))
      return; // No work to do

    auxNRows /= count;
    unsigned int aux = (mod + mRows * minionId) / auxNRows;
    offsetNum += jump * aux;
    unsigned int initialAddrIn = ((mod + mRows * minionId) - aux * auxNRows) * actPitch[dimRow];

    unsigned int offsetIn[dstDimNum], offsetOut[dstDimNum];
    unsigned int initialAddr = offsetNum;
    getCoordinates(offsetIn, initialAddrIn, dstDimNum, actPitch);
    getCoordinates(offsetOut, initialAddr, dstDimNum, dstPitch);

    unsigned int addrOut = 0;
    for (int i = lastDim; i >= 0; i--) {
      offsetOut[i] += offsetIn[i];
      addrOut += dstPitch[i] * offsetOut[i];
    }
    bool done = false;
    while (mRows > 0) {
      insertRow<srcType>((uint8_t *) dst, (uint8_t *) src2, addrOut, 
                initialAddrIn, typeSize, lanes, res, gatherValues);
      for (int j = dimRow; j >= 0; j--) {
        if (likely(offsetIn[j] != (actIndex[j] - 1))) {
          initialAddrIn += actPitch[j];
          addrOut += dstPitch[j];
          offsetIn[j]++;
          break;
        } else if (likely(j != 0)){
          initialAddrIn -= (actIndex[j] - 1) * actPitch[j];
          addrOut -= (actIndex[j] - 1) * dstPitch[j];
          offsetIn[j] = 0;
        } else {
          initialAddrIn = offsetIn[j]  = 0;
          offsetNum += jump;
          addrOut = offsetNum;
        }
      }
      mRows--;
    }
  } else {
    unsigned int auxNRows = actIndex[0];
    for (int i = 1; i < dstDimNum - 1; i++)
      auxNRows *= actIndex[i];

    if (auxNRows > activeMinions) {
      unsigned int mRows = auxNRows / activeMinions; 
      unsigned int mod = auxNRows - activeMinions * mRows;
      unsigned int initialAddrIn;
      // We add to the initial address the new address in the tensor
      if (minionId < mod) {
        ++mRows;
        initialAddrIn = mRows * actPitch[dimRow] * minionId;
      } else
        initialAddrIn = (mod + minionId * mRows) * actPitch[dimRow];
      unsigned int k, offsetIn[dstDimNum], offsetOut[dstDimNum];
      getNonPaddingCoordinates(offsetIn, initialAddrIn, dstDimNum, actPitch,
                               actIndex, k);
      getNonPaddingCoordinates(offsetOut, offsetNum, dstDimNum, dstPitch,
                               dstIndex, k);
      unsigned int addrOut = 0;
      for (int i = dstDimNum - 1; i >= 0; i--) {
        offsetOut[i] += offsetIn[i];
        addrOut += dstPitch[i] * offsetOut[i];
      }
      for (int i = 0; i < mRows; i++) {
        for (int j = 0; j < count; j++) {
          insertRow<srcType>((uint8_t *) dst, (uint8_t *) src2, addrOut, 
                    initialAddrIn, typeSize, lanes, res, gatherValues);
          addrOut += actIndex[axis] * dstPitch[axis];
        }
        addrOut -= count * actIndex[axis] * dstPitch[axis];
        for (int j = dimRow; j >= 0; j--) {
          if (offsetIn[j] != (actIndex[j] - 1)) {
            addrOut += dstPitch[j];
            initialAddrIn += actPitch[j];
            offsetIn[j]++;
            break;
          } else {
            addrOut -= (actIndex[j] - 1) * dstPitch[j];
            initialAddrIn -= (actIndex[j] - 1) * actPitch[j];
            offsetIn[j] = 0;
          }
        }
      }
    } else {
      unsigned int mperRow = activeMinions / auxNRows; 
      if (minionId >= mperRow * auxNRows)
        return;                                   
      unsigned int rowtomin = minionId / mperRow; 

      unsigned int offsetOut[dstDimNum];
      for (unsigned int i = 0; i < dstDimNum; i++) {
        offsetOut[i] = coord[i];
      }

      if(axis > 0) {
        unsigned int falsepitch[axis];
        falsepitch[dimRow] = 1;
        for (int i = dimRow; i > 0; i--)
          falsepitch[i - 1] = falsepitch[i] * actIndex[i];
        
        for (int i = 0; i < axis; i++) {
          unsigned int aux = rowtomin / falsepitch[i];
          offsetOut[i] += aux;
          rowtomin -= aux * falsepitch[i];
        }
      }
      unsigned int addrOut = 0; 
      for (int i = axis; i >= 0; i--) {
        addrOut += dstPitch[i] * offsetOut[i];
      }
      unsigned int lastRowElem = addrOut + actIndex[axis] * dstPitch[axis] * count;
      unsigned int save = addrOut;
      unsigned int cll = 64 / getsize<srcType>();
      unsigned int modulo = addrOut % cll;
      //unsigned int maximalPos = jump * count;
      unsigned int clperRow = (modulo + (jump * count) - 1) / cll + 1;
      unsigned int mcl = clperRow / mperRow; 
      unsigned int mod = clperRow - mperRow * mcl;
      unsigned int maxRead;
      unsigned int minmodule = minionId % mperRow;
      if (minmodule != 0) {
        addrOut -= modulo;
        if (minmodule < mod){
          ++mcl;
          addrOut += mcl * cll * minmodule;
        } else {
          addrOut += (mod + minmodule * mcl) * cll;
        }
        maxRead = mcl * cll;
      } else {
        if (mod != 0) {
          ++mcl;
        }
        maxRead = mcl * cll - modulo;
      }
      if (mcl == 0) {
        return; 
      }
      //maximalPos += save - 1;
      unsigned int k;
      getNonPaddingCoordinates(offsetOut, addrOut, dstDimNum, dstPitch,
                               dstIndex, k);
      addrOut = 0;
      for (unsigned int i = 0; i < dstDimNum; i++) {
        addrOut += offsetOut[i] * dstPitch[i];
      }

      unsigned int offsetIn[dstDimNum];
      for (unsigned int i = 0; i < dstDimNum; i++) {
        offsetIn[i] = offsetOut[i] - coord[i];
      }
      offsetIn[axis] = offsetIn[axis] % actIndex[axis];
      unsigned int initialAddrIn = 0;
      for (unsigned int i = 0; i < dstDimNum; i++) {
        initialAddrIn += offsetIn[i] * actPitch[i];
      }
      maxRead = std::min(maxRead, lastRowElem - addrOut); 
      unsigned int length = std::min(maxRead, actIndex[axis] - offsetIn[axis]);
      int auxlanes, auxres;
      getLanesResTView<srcType>(auxlanes, auxres, length);
      maxRead -= length;
      mask = (1 << (((auxlanes - 1) % 8) + 1)) - 1;
      __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      mask = (1 << auxres) - 1;
      __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));

      insertRow<srcType>((uint8_t *) dst, (uint8_t *) src2, addrOut, 
                initialAddrIn, typeSize, auxlanes, auxres, gatherValues);
      addrOut += length * dstPitch[axis];
      initialAddrIn -= offsetIn[axis] * actPitch[axis];
      
      mask = (1 << (((lanes - 1) % 8) + 1)) - 1;
      __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      mask = (1 << res) - 1;
      __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      while (maxRead > actIndex[axis]) {
        insertRow<srcType>((uint8_t *) dst, (uint8_t *) src2, addrOut, 
                  initialAddrIn, typeSize, lanes, res, gatherValues);
        maxRead -= actIndex[axis];
        addrOut += actIndex[axis] * dstPitch[axis];
      }
      getLanesResTView<srcType>(auxlanes, auxres, maxRead);
      mask = (1 << (((auxlanes - 1) % 8) + 1)) - 1;
      __asm__ __volatile__ ("mov.m.x m1, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      mask = (1 << auxres) - 1;
      __asm__ __volatile__ ("mov.m.x m2, %[mask], 0x0 \n"
                            :
                            : [ mask ] "r" (mask));
      insertRow<srcType>((uint8_t *) dst, (uint8_t *) src2, addrOut, 
                initialAddrIn, typeSize, auxlanes, auxres, gatherValues);
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibExtractTensorInst(void *dst, void *dstDims,
                                      void *dstPitches, unsigned int dstDimNum,
                                      void *src, void *srcDims,
                                      void *srcPitches, void *pcoord,
                                      float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int *coord = (unsigned int *)pcoord;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eOffsets[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < dstDimNum; i++) {
    eDims[i] = dstIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
    eOffsets[i] = coord[i];
  }

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              tOutput[x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                      w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5]] =
                  tInput[(eOffsets[0] + x) * eSrcPitch[0] +
                         (eOffsets[1] + y) * eSrcPitch[1] +
                         (eOffsets[2] + z) * eSrcPitch[2] +
                         (eOffsets[3] + w) * eSrcPitch[3] +
                         (eOffsets[4] + q) * eSrcPitch[4] +
                         (eOffsets[5] + r) * eSrcPitch[5]];
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibExtractTensorInstThreaded(void *dst, void *dstDims,
                                              void *dstPitches,
                                              unsigned int dstDimNum, void *src,
                                              void *srcDims, void *srcPitches,
                                              void *pcoord, float *scale,
                                              int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int *coord = (unsigned int *)pcoord;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialaddrOut, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialaddrOut, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coordOut[dstDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coordOut, initialaddrOut, dstDimNum, dstPitch,
                           dstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++)
    offsetOut += dstPitch[i] * coordOut[i];
  unsigned int offsetIn = 0;
  for (unsigned int i = 0; i < dstDimNum; ++i)
    offsetIn += (coord[i] + coordOut[i]) * srcPitch[i];

  unsigned int posMaxOut = maxRead + initialaddrOut;
  bool done = false;
  while (!done && (offsetOut < posMaxOut)) {
    tOutput[offsetOut] = tInput[offsetIn];
    done = getOffsets(dstDimNum, coordOut, offsetIn, offsetOut, dstIndex,
                      srcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialaddrOut, clperminion);
}

template <typename srcType, typename indexType>
void dnn_lib::fwdLibGatherInst(void *dstT, void *dstDims, void *dstPitches,
                               void *srcT, void *srcDims, void *srcPitches,
                               unsigned int srcDimsNum, void *indexT,
                               void *indicesDims, void *pindicesPitches,
                               unsigned int batchedDims, float *scale,
                               int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  const Addresser<indexType> tIndices(indexT, scale[1], offset[1]);

  unsigned int *srcIndex = (unsigned int *)srcDims;
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *indicesIndex = (unsigned int *)indicesDims;

  unsigned int *srcPitch = (unsigned int *)srcPitches;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *indicesPitch = (unsigned int *)pindicesPitches;

  size_t index;
  uint64_t srcAddr;
  uint64_t srcAddrUp;
  uint64_t dstAddr;
  auto val = tInput[0];
  // For each sample in the batch:
  for (size_t i = 0; i < dstIndex[0]; i++) {
    // For each slice (small fragment) that we copy from the source memory:
    for (size_t j = 0; j < dstIndex[1]; j++) {
      // Reads index [i,j]
      if (batchedDims != srcDimsNum - 1) {
        index = tIndices[i * indicesPitch[batchedDims] + j];
        srcAddr = index * srcPitch[batchedDims];
        srcAddrUp = (index + 1) * srcPitch[batchedDims];
        dstAddr = i * dstPitch[batchedDims] + j * dstPitch[batchedDims + 1];
      } else {
        index = tIndices[j];
        srcAddr = i * srcPitch[batchedDims - 1] + index * srcPitch[batchedDims];
        srcAddrUp =
            i * srcPitch[batchedDims - 1] + (index + 1) * srcPitch[batchedDims];
        dstAddr = i * dstPitch[batchedDims - 1] + j;
      }
      // perform the copy
      for (uint64_t i = srcAddr, num = 0; i < srcAddrUp; i++, num++) {
        val = tInput[i];
        tOutput[dstAddr + num] = val;
      }
    }
  }
}

// The threaded version of the GatherInst function generalises the function to
// any given dimensions for the two source tensors (tInput and tIndices). tInput
// dims: d0 x ··· x dn. batchedDims: i (number between 0 and n). tIndices dims:
// i0 x ··· x ik. Then, the tOutput tensor dimensions are determined. tOutput
// dims: d0 x ··· x d(i-1) x i0 x ··· x ik x d(i+1) x ··· x dn (dstDimsNum =
// srcDimsNum + indicesDimsNum - 1). The GatherInst function consists in copying
// the source tensor's elements in the following way:
// tOutput(x0,...,x(i-1),y0,...,yk,x(i+1),...,xn) =
// tInput(x0,...,x(i-1),tIndices(y0,...,yk),x(i+1),...,xn). The elements in the
// tIndices tensor must be integers between 0 and di - 1, so they are valid
// index values for the i-th dimension of the source tensor tInput.

template <typename srcType, typename indexType>
void dnn_lib::fwdLibGatherInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT, void *srcDims,
    void *srcPitches, unsigned int srcDimsNum, void *indexT, void *indicesDims,
    void *pindicesPitches, unsigned int indicesDimsNum,
    unsigned int batchedDims, // indicesDimsNum is an new parameter for the
                              // threaded version.
    float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  const Addresser<indexType> tIndices(indexT, scale[1], offset[1]);

  unsigned int *srcIndex = (unsigned int *)srcDims;
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *indicesIndex = (unsigned int *)indicesDims;

  unsigned int *srcPitch = (unsigned int *)srcPitches;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *indicesPitch = (unsigned int *)pindicesPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int dstDimsNum = srcDimsNum + indicesDimsNum - 1;
  unsigned int coordOut[dstDimsNum];
  unsigned int last_non_zero_coord;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimsNum, dstPitch,
                           dstIndex, last_non_zero_coord);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < last_non_zero_coord; i++)
    offsetOut += dstPitch[i] * coordOut[i];
  unsigned int offsetIndices = 0;
  for (unsigned int i = 0; i < indicesDimsNum; i++)
    offsetIndices += coordOut[batchedDims + i] * indicesPitch[i];

  unsigned int offsetIn = 0;
  for (unsigned int i = 0; i < batchedDims; i++)
    offsetIn += srcPitch[i] * coordOut[i];
  unsigned int index = tIndices[offsetIndices];
  offsetIn += srcPitch[batchedDims] * index;
  for (unsigned int i = batchedDims + 1; i < srcDimsNum; i++)
    offsetIn +=
        srcPitch[i] *
        coordOut[indicesDimsNum + i - 1]; // could iterate just until the last
                                          // source non-zero coordInate. todo

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    tOutput[offsetOut] = tInput[offsetIn];
    // Coordinates are updated to the next position that must be copied.
    for (int j = dstDimsNum - 1; j >= 0; j--) {
      if (coordOut[j] != (dstIndex[j] - 1)) {
        offsetOut += dstPitch[j];
        coordOut[j]++;
        if (j >= batchedDims + indicesDimsNum)
          offsetIn += srcPitch[j - indicesDimsNum + 1];
        else if (batchedDims <= j) {
          offsetIndices += indicesPitch[j - batchedDims];
          offsetIn += (tIndices[offsetIndices] - index) * srcPitch[batchedDims];
          index = tIndices[offsetIndices];
        } else
          offsetIn += srcPitch[j];
        break; // Once the coordinates have been updated, a new copy can be
               // performed.
      } else if (j != 0) {
        offsetOut -= (dstIndex[j] - 1) * dstPitch[j];
        coordOut[j] = 0;
        if (j >= batchedDims + indicesDimsNum) {
          unsigned int k = j - indicesDimsNum + 1;
          offsetIn -= (srcIndex[k] - 1) * srcPitch[k];
        } else if (batchedDims <= j) {
          offsetIndices -= (indicesIndex[j - batchedDims] - 1) *
                           indicesPitch[j - batchedDims];
          offsetIn += (tIndices[offsetIndices] - index) * srcPitch[batchedDims];
          index = tIndices[offsetIndices];
        } else
          offsetIn += srcPitch[j];
      } else
        done = true; // The end of the destination tensor has been reached.
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}


template <typename srcType, typename indexType>
void dnn_lib::fwdLibGatherRangesInst(
    void *dstT, void *dstDims, void *dstPitches, void *dst2T, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimsNum, void *prangesT, void *prangesDims,
    void *prangesPitches, float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstT, scale[3], offset[3]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  Addresser<indexType> tRanges(prangesT, scale[1], offset[1]);
  Addresser<indexType> tLengths(dst2T, scale[2], offset[2]);

  unsigned int *srcIndex = (unsigned int *)srcDims;
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *rangesIndex = (unsigned int *)prangesDims;
  unsigned int *lenIndex = (unsigned int *)dst2Dims;

  unsigned int *srcPitch = (unsigned int *)srcPitches;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *rangesPitch = (unsigned int *)prangesPitches;
  unsigned int *lenPitch = (unsigned int *)dst2Pitches;

  // Offset into the output tensor that keeps track of where to start
  // copying data.
  uint64_t outP = 0;

  // unsigned dataElementSize = dataTy.getElementSize();
  indexType numExamples = rangesIndex[0];
  indexType exampleSize = rangesIndex[1];

  // Keep track of the total number of elements gathered across all
  // examples for a sanity check later.
  size_t grandTotalLen = 0;

  // For each example in ranges:
  for (size_t example = 0; example < numExamples; example++) {
    // Keep a running total of the lengths of all ranges in this example
    // to record into lengthsT once the entire example is processed.
    indexType totalLen = 0;

    // For each range in the example:
    for (indexType range = 0; range < exampleSize; range++) {
      // Get the start index and range length.
      indexType startIdx =
          tRanges[example * rangesPitch[0] + range * rangesPitch[1]];
      indexType len = tRanges[example * rangesPitch[0] +
                              range * rangesPitch[1] + 1 * rangesPitch[2]];

      // Add the length of this current range to the example length counter.
      totalLen += len;

      // Copy the specified data to outT.
      uint64_t srcAddr = startIdx * srcPitch[0];
      uint64_t srcAddrUp = (startIdx + len) * srcPitch[0];

      auto val = tInput[0];
      for (uint64_t i = srcAddr, j = 0; i < srcAddrUp; i++, j++) {
        val = tInput[i];
        tOutput[outP + j] = val;
      }

      // Advance the offset into outT.
      outP += len * dstPitch[0];
    }

    // Record the total number of elements gathered for the example in
    // lengthsT.
    tLengths[example * lenPitch[0]] = totalLen;

    // Add the total length of the entire example to the grand total.
    grandTotalLen += static_cast<size_t>(totalLen);
  }

  // Make sure that number of elements written to outT is equal to the
  // total of all elements in lengthsT.
  // assert(grandTotalLen == (outP / dstPitch[0]));
}


// The range tensor has dimensions n x m x 2, where n is the number of examples and m is the number of
// ranges per example. For any pair (i,j), the element ranges[i,j,0] is the source tensor batch number from
// which the copy will start, and the element ranges[i,j,1] is the length of the copy, that is, the amount
// of batches of the source tensor that will be copied.

template <typename srcType, typename indexType>
void dnn_lib::fwdLibGatherRangesInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *dst2T, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimsNum, void *prangesT, void *prangesDims,
    void *prangesPitches, float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  Addresser<srcType> tOutput(dstT, scale[3], offset[3]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  Addresser<indexType> tRanges(prangesT, scale[1], offset[1]);
  Addresser<indexType> tLengths(dst2T, scale[2], offset[2]);

  unsigned int *srcIndex = (unsigned int *)srcDims;
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *rangesIndex = (unsigned int *)prangesDims;
  unsigned int *lenIndex = (unsigned int *)dst2Dims;

  unsigned int *srcPitch = (unsigned int *)srcPitches;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *rangesPitch = (unsigned int *)prangesPitches;
  unsigned int *lenPitch = (unsigned int *)dst2Pitches;

  unsigned int last_minion = activeMinions - 1;
  
  if (minionId < last_minion) {

    unsigned int numElemsDst = dstPitch[0]*dstIndex[0];  
    unsigned int initialAddr, maxRead;
    size_t typeSize = getsize<srcType>();
    getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions - 1); 
    if (maxRead == 0)
      return;
 
    // Assumption: srcDimsNum = dstDimsNum.
    unsigned int coordOut[srcDimsNum];
    unsigned int last_non_zero_coord;
    getNonPaddingCoordinates(coordOut, initialAddr, srcDimsNum, dstPitch, dstIndex, last_non_zero_coord);
    
    uint64_t offsetOut = 0;
    for (unsigned int i = 0; i < last_non_zero_coord; i++) {
      offsetOut += dstPitch[i]*coordOut[i];
    }
    
    uint64_t offsetRanges = rangesPitch[2];
    indexType length = tRanges[offsetRanges];
    unsigned int range = 0;
    unsigned int accumLength = 0;
    unsigned int exampleSize = rangesIndex[1];
    unsigned int exampleMem = rangesIndex[1]*rangesPitch[1];
    while ((accumLength + length)*dstPitch[0] < offsetOut) {
      accumLength += length;
      offsetRanges += rangesPitch[1];
      length = tRanges[offsetRanges];
      range++;
      if (range == exampleSize) {
        offsetRanges += rangesPitch[0] - exampleMem;
        range = 0;
      }
    }
    offsetRanges -= rangesPitch[2];

    uint64_t offsetIn = tRanges[offsetRanges]*srcPitch[0]; // tRanges[offsetRanges] is the starting batch id.
    unsigned int count = 1;
    while ((accumLength + count)*dstPitch[0] < offsetOut) {
      offsetIn += srcPitch[0];
      count++;
    }
    count--;
    unsigned int positionInBatch = offsetOut - (accumLength + count)*dstPitch[0];
    offsetIn += positionInBatch;
    unsigned int coordIn[srcDimsNum];
    getNonPaddingCoordinates(coordIn, offsetIn, srcDimsNum, srcPitch, srcIndex, last_non_zero_coord); // useless last parameter.
    
    unsigned int batchElems = 1;
    for (int i = 1; i < srcDimsNum; ++i) batchElems *= srcIndex[i]; // avoiding padding elements.

    unsigned int posMax = maxRead + initialAddr;  
    bool done = false;
    bool doneIn = false; // useful for skipping padding positions in the source tensor.
    while (!done && (offsetOut < posMax)) {
      tOutput[offsetOut] = tInput[offsetIn];
      done = getOffsets(srcDimsNum, coordOut, offsetOut, dstIndex, dstPitch);
      positionInBatch++;
      if (positionInBatch != batchElems) doneIn = getOffsets(srcDimsNum, coordIn, offsetIn, srcIndex, srcPitch);
      else {
        positionInBatch = 0;
        count++;
        if (count != length) doneIn = getOffsets(srcDimsNum, coordIn, offsetIn, srcIndex, srcPitch);
        else {
          count = 0;
          ++range;
          if (range != exampleSize) offsetRanges += rangesPitch[1];
          else {
            range = 0;
            offsetRanges += rangesPitch[0] - (exampleSize - 1)*rangesPitch[1];
          }
          offsetIn = tRanges[offsetRanges];
          length = tRanges[offsetRanges + rangesPitch[2]];
        }
      }
    }

    if (!DO_EVICTS) return;
    unsigned int clperminion = maxRead*sizeof(srcType)/64;
    if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
  }

// For coherence reasons only one minion should be able to write on the Length tensor, since in practice
// it will just be a short vector (not more than a couple cl's). This implementation involves the last
// active minion.

  else if (minionId == last_minion) {
    unsigned int numExamples = rangesIndex[0];
    unsigned int exampleSize = rangesIndex[1];
    unsigned int offsetRanges = rangesPitch[2];
    unsigned int auxoffsetRanges = rangesPitch[2]; // this aux variable helps avoiding products.
    unsigned int offsetLengths = 0;
    for (size_t example = 0; example < numExamples; example++) { // size_t or indexType?
      indexType totalLength = 0;
      for (size_t range = 0; range < exampleSize; range++) {
          totalLength += tRanges[offsetRanges];
          offsetRanges += rangesPitch[1];
      }
      tLengths[offsetLengths] = totalLength;
      offsetRanges = auxoffsetRanges + rangesPitch[0];
      auxoffsetRanges = offsetRanges;
      offsetLengths += lenPitch[0];
    }
/*
    // Todo: initialAddr should be the virtual address of the Length tensor.
    if (!DO_EVICTS) return;
    unsigned int clperminion = lenIndex[0]*lenPitch[0]*sizeof(srcType)/64;
    if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
*/
  }
}




template <typename srcType>
void dnn_lib::fwdLibScatterAssignInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *indexT,
                                      void *indicesDims, void *pindicesPitches,
                                      void *slicesT, void *slicesDims,
                                      void *slicesPitches, float *scale,
                                      int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> tSlices(slicesT, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);

  long long *tIndices = (long long *)indexT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *indicesIndex = (unsigned int *)indicesDims;
  unsigned int *slicesIndex = (unsigned int *)slicesDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *indicesPitch = (unsigned int *)pindicesPitches;
  unsigned int *slicesPitch = (unsigned int *)slicesPitches;

  // For each slice (small fragment) that we copy from the source memory:
  uint64_t n = slicesPitch[0];
  auto val = tSlices[0];
  /*for (int i = 0; i < indicesIndex[0]; i++){ 
   val =  (int)tIndices[i];

   tOutput[i] = val;
  }
  return; */
  for (int j = 0; j < indicesIndex[0]; j++) {
    // Reads index [j]
    long long index = tIndices[j];
    // std::copy(&tSlices[j*slicesPitch[0]], &tSlices[j*slicesPitch[0]] +
    // slicesPitch[0], &tOutput[index*dstPitch[0]]);
    uint64_t srcAddr = j * n;
    uint64_t dstAddr = index * dstPitch[0];
    // perform the copy
    for (uint64_t i = 0; i < n; i++) {
      val = tSlices[srcAddr + i];
      tOutput[dstAddr + i] = val;
    }
  }
}


template <typename srcType>
void dnn_lib::fwdLibScatterAssignInstThreaded(void *dstT, void *dstDims,
                                              void *dstPitches, unsigned int dstDimNum, void *indexT,
                                              void *indicesDims, void *pindicesPitches,
                                              void *slicesT, void *slicesDims,
                                              void *slicesPitches, float *scale,
                                              int32_t *offset, uint64_t flags) {


  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;
   
  const Addresser<srcType> tSlices(slicesT, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);

  long long *tIndices = (long long *)indexT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *indicesIndex = (unsigned int *)indicesDims;
  unsigned int *slicesIndex = (unsigned int *)slicesDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *indicesPitch = (unsigned int *)pindicesPitches;
  unsigned int *slicesPitch = (unsigned int *)slicesPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[dstDimNum];
  for (unsigned int i = 0; i < dstDimNum; i++)
    coord[i] = 0;
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetIn = 0; // Doesn't include slicesPitch[0]. offsetIn doesn't
                             // have the conventional meaning
  unsigned int offsetOut = dstPitch[0] * coord[0];

  unsigned int slicesPitch_0 = slicesPitch[0];
  slicesPitch[0] = 0;

  for (unsigned int j = 1; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
    offsetIn += slicesPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  bool change = true;
  int offset0 = 0;
  while (!done && (offsetOut < posMax)) {
    if (change){
      offset0 = indicesIndex[0] - 1;
      while(offset0 >= 0){
        if (tIndices[offset0] == coord[0]) break;
        offset0--;
      }
      change = false;
    }
    if (offset0 >= 0) tOutput[offsetOut] = tSlices[offsetIn + offset0*slicesPitch_0];
    

    for (int j = dstDimNum - 1; j >= 0; j--) {
      if (coord[j] != (dstIndex[j] - 1)) {
        offsetOut += dstPitch[j];
        offsetIn += slicesPitch[j];
        coord[j]++;
        if (j == 0) change = true;
        break;
      } else if (j != 0) {
        offsetOut -= (dstIndex[j] - 1) * dstPitch[j];
        offsetIn -= (slicesIndex[j] - 1) * slicesPitch[j];
        coord[j] = 0;
      } else {
        done = true;
        break;
        }
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}



template <typename srcType>
void dnn_lib::fwdLibBatchOneHotInst(void *pdst, void *pdstDims,
                                    void *pdstPitches, void *pdata,
                                    void *pdataDims, void *pdataPitches,
                                    void *pvalues, void *pvaluesDims,
                                    void *pvaluesPitches, void *plengths,
                                    float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  const Addresser<srcType> tValues(pvalues, scale[1], offset[1]);
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *valuesIndex = (unsigned int *)pvaluesDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *valuesPitch = (unsigned int *)pvaluesPitches;

  auto batchSize = dataIndex[0];
  auto featureCnt = dataIndex[1];

  for (size_t batchId = 0; batchId < batchSize; batchId++) {
    size_t offset = 0;
    for (size_t featureId = 0; featureId < featureCnt; featureId++) {
      auto curValue = tAInput[batchId * dataPitch[0] + featureId];
      auto curLength = lengths[featureId];
      for (size_t i = offset, e = offset + curLength; i != e; i++) {
        int64_t dstAddr = batchId * dstPitch[0] + i;
        if (curValue == tValues[i]) {
          tOutput[dstAddr] = (float)1;
        } else {
          tOutput[dstAddr] = (float)0;
        }
      }
      offset += curLength;
    }
    // assert(offset == dstIndex[1] && "Sum of Lengths must be equal to size of
    // Values");
  }
}

template <typename srcType>
void dnn_lib::fwdLibBatchOneHotInstThreaded(void *pdst, void *pdstDims,
                                            void *pdstPitches, void *pdata,
                                            void *pdataDims, void *pdataPitches,
                                            void *pvalues, void *pvaluesDims,
                                            void *pvaluesPitches, void *plengths,
                                            float *scale, int32_t *offset, uint64_t flags) {


  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  const Addresser<srcType> tValues(pvalues, scale[1], offset[1]);
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *valuesIndex = (unsigned int *)pvaluesDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *valuesPitch = (unsigned int *)pvaluesPitches;

  auto batchSize = dataIndex[0];
  auto featureCnt = dataIndex[1];

  unsigned int numElemsDst = batchSize * dstPitch[0]; // Total number of elements in the tensor



  unsigned int dstAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, dstAddr, maxRead,
                        minionId, activeMinions);

  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position

  unsigned int k;          // Amount of non-zero coordinates
  unsigned int coord[2]; // Vector of coordinates



  getNonPaddingCoordinates(coord, dstAddr, 2, dstPitch, dstIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int batchId = offsetOut/dstPitch[0];
  unsigned int i = offsetOut - batchId * dstPitch[0]; 
  unsigned int l = i;
  unsigned int featureId = 0;
  while (l >= lengths[featureId]) {
    l -= lengths[featureId];
    featureId++;
    if (featureId >= featureCnt)  {
      featureId = 0;
      l = i = 0;
      batchId++;
      break;
    }
  }

  
  unsigned int posMax = maxRead + offsetOut;


  bool done = false;
  bool minionEnd = false;

  while (!done && !minionEnd) {
    while (batchId < batchSize){
      size_t offset = i;
      while (featureId < featureCnt){
        auto curValue = tAInput[batchId * dataPitch[0] + featureId];
        auto curLength = lengths[featureId];
        while (i < offset + curLength) {
          if (curValue == tValues[i]) {
            tOutput[offsetOut] = (float)1;
          } else {
            tOutput[offsetOut] = (float)0;
          }
          offsetOut++;
          if (offsetOut > posMax) {
            minionEnd = true;
            break;
          } 
          i++;
        }
        offset += curLength;
        featureId++;
        if (minionEnd == true) 
          break;
      }
      i = 0;
      featureId = 0;
      batchId++;
      offsetOut += dstPitch[0];
      if (offsetOut > posMax) {
       minionEnd = true;
      }
      if (minionEnd == true) 
        break;
    }
    if (batchId == batchSize)
      done = true; 
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*dstAddr, clperminion);
}

//===----------------------------------------------------------------------===//
//                      Local Response Normalization
//===----------------------------------------------------------------------===//

template <typename srcType>
void dnn_lib::fwdLibLocalResponseNormalizationInst(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    unsigned int halfWindowSize, float alpha, float beta, float k, float *scale,
    int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
  Addresser<srcType> tScale(dst2Matrix, scale[2], offset[2]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *dst2Index = (unsigned int *)dst2MatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *dst2Pitch = (unsigned int *)dst2MatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;

  // LRN node does not change the shape of the input.
  // assert((dstIndex[0] == actIndex[0]) && (dstIndex[1] == actIndex[1]) &&
  // (dstIndex[2] == actIndex[2]) && (dstIndex[3] == actIndex[3]) && "Output of
  // LRN node must be same shape as input");

  // LRN node normalizes across channels, so the input must have a minimum
  // depth of 1.
  // assert(actIndex[3] > 0 && "Input of LRN node must have a minimum depth of
  // 1");

  auto windowSize = 2 * halfWindowSize + 1;
  float inversedWindowSize;
  fpReciprocalSingleElement(windowSize, inversedWindowSize);
  float normedAlpha = alpha * inversedWindowSize;

  // For every input in the batch:
  for (size_t n = 0; n < actIndex[0]; n++) {

    // For every row:
    for (size_t h = 0; h < actIndex[1]; h++) {

      // For every column:
      for (size_t w = 0; w < actIndex[2]; w++) {

        // For every channel:
        for (size_t c = 0; c < actIndex[3]; c++) {
          auto squareSum = tAInput[0];
          squareSum = 0.0;
          for (size_t i = (c >= halfWindowSize ? c - halfWindowSize : 0);
               i <=
               std::min(c + halfWindowSize, (long unsigned int)actIndex[3] - 1);
               i++) {
            auto val = tAInput[n * actPitch[0] + h * actPitch[1] +
                               w * actPitch[2] + i * actPitch[3]];
            squareSum += val * val;
          }

          auto scale = k + normedAlpha * squareSum;

          // This will be used to accelerate the backward pass.
          tScale[n * dst2Pitch[0] + h * dst2Pitch[1] + w * dst2Pitch[2] +
                 c * dst2Pitch[3]] = scale;

          auto normFactor = getPow(scale, -beta);
          auto op = tAInput[n * actPitch[0] + h * actPitch[1] +
                            w * actPitch[2] + c * actPitch[3]];
          op *= normFactor;
          tOutput[n * dstPitch[0] + h * dstPitch[1] + w * dstPitch[2] +
                  c * dstPitch[3]] = op;
        }
      }
    }
  }
}

// First threaded version, assuming dstMatrix and dst2Matrix have the same
// Pitches. Without this assumption, coherence might be lost and therefore this
// version is not correct. Notice that dst2Matrix is only needed for backward
// pass, i.e. ETSOC won't be using it. Actually, we could skip generating it.

template <typename srcType>
void dnn_lib::fwdLibLocalResponseNormalizationInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    unsigned int halfWindowSize, float alpha, float beta, float k, float *scale,
    int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
  Addresser<srcType> tScale(dst2Matrix, scale[2], offset[2]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *dst2Index = (unsigned int *)dst2MatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *dst2Pitch = (unsigned int *)dst2MatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;

  // LRN node does not change the shape of the input.
  // assert((dstIndex[0] == actIndex[0]) && (dstIndex[1] == actIndex[1]) &&
  // (dstIndex[2] == actIndex[2]) && (dstIndex[3] == actIndex[3]) && "Output of
  // LRN node must be same shape as input");

  // LRN node normalizes across channels, so the input must have a minimum
  // depth of 1.
  // assert(actIndex[3] > 0 && "Input of LRN node must have a minimum depth of
  // 1");

  auto windowSize = 2 * halfWindowSize + 1;
  float inversedWindowSize;
  fpReciprocalSingleElement(windowSize, inversedWindowSize);
  float normedAlpha = alpha * inversedWindowSize;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  const unsigned int srcDimNum = 4;
  unsigned int coord[srcDimNum] = {0, 0, 0, 0};
  unsigned int t = 0;  //this variable is usually called k, but n this case the name k is already used
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           t);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < t; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    auto squareSum = tAInput[offsetIn];
    squareSum = 0.0;
    size_t c = size_t(coord[3]);
    for (unsigned int i = (c >= halfWindowSize ? c - halfWindowSize : 0);
         i <= std::min(c + halfWindowSize, (long unsigned int)actIndex[3] - 1);
         i++) {
      auto val = tAInput[offsetIn + size_t((i - c) * actPitch[3])];
      squareSum += val * val;
    }

    auto scale = k + normedAlpha * squareSum;

    // This will be used to accelerate the backward pass.
    tScale[offsetOut] = scale;

    auto normFactor = getPow(scale, -beta);
    auto op = tAInput[offsetIn];
    op *= normFactor;
    tOutput[offsetOut] = op;

    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, dstIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}


template <typename srcType>
void dnn_lib::fwdLibLocalResponseNormalizationInstVectorized(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *dst2Matrix, void *dst2MatrixDims, void *dst2MatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    unsigned int halfWindowSize, float alpha, float beta, float k, float *scale,
    int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  uintptr_t srcAddr = (uintptr_t)activations;
  Addresser<srcType> tOutput(dstMatrix, scale[1], offset[1]);
  Addresser<srcType> tScale(dst2Matrix, scale[2], offset[2]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *dst2Index = (unsigned int *)dst2MatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *dst2Pitch = (unsigned int *)dst2MatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;

  auto windowSize = 2 * halfWindowSize + 1;
  float inversedWindowSize;
  fpReciprocalSingleElement(windowSize, inversedWindowSize);
  float normedAlpha = alpha * inversedWindowSize;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  const unsigned int srcDimNum = 4;
  unsigned int coord[srcDimNum] = {0, 0, 0, 0};
  unsigned int t = 0;  //this variable is usually called k, but n this case the name k is already used
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           t);
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < t; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int mask;
  while (!done && (offsetOut < posMax)) {
    float squareSum = 0.0;
    size_t c = size_t(coord[3]);
    unsigned int start = (c >= halfWindowSize ? c - halfWindowSize : 0);
    unsigned int end = std::min(c + halfWindowSize, (long unsigned int)actIndex[3] - 1);
    unsigned int registers = (end - start + 1)/8;
    unsigned int mod = (end - start + 1) - 8*registers;
    constexpr uint32_t offs = 32;
    srcAddr += (offsetIn + (start - c)*actPitch[3]) * typeSize;
     
    mask = ((1 << mod) - 1);     
    __asm__ __volatile__("mov.m.x m1, %[mask], 0 \n" 
                         "mov.m.x m0, zero, 0xff \n"
                         "fxor.pi f0, f0, f0\n"
                         "add t0, zero, zero\n"
                     
                         
                         "ble %[registers], t0, 2f\n"
                         "1:\n"
                         "flw.ps f1, 0x0(%[src])\n"
                         "fmul.ps f1, f1, f1\n"
                         "fadd.ps f0, f0, f1\n"
                         "addi t0, t0, 0x1\n"
                         "addi %[src], %[src], %[offs]\n"
                         "blt t0, %[registers], 1b\n"
                         "2:\n"
                         "ble %[mod], zero, 3f\n"
                         "maskand m0, m1, m0 \n"
                         "flw.ps f1, 0x0(%[src])\n"
                         "fmul.ps f1, f1, f1\n"
                         "fadd.ps f0, f0, f1\n"
                         "3:\n"
                         "mov.m.x m0, zero, 0xff \n"
                         "fswizz.ps f30, f0, 0xe \n"
                         "fadd.ps f0,f30, f0 \n"
                         "fswizz.ps f30, f0, 0x1 \n"
                         "fadd.ps f0,f30, f0 \n"
                         "fmvs.x.ps %[sum], f0, 0x4 \n"
                         "fmv.w.x f30, %[sum] \n"
                         "fadd.s f0, f30, f0 \n"
                         "fmvs.x.ps %[sum], f0, 0x0 \n"

                         : [ sum ] "+r"(squareSum)
                         : [ mask ] "r"(mask),
                           [ mod ] "r"(mod),
                           [ offs ] "I"(offs),
                           [ src ] "r"(srcAddr),
                           [ registers ] "r"(registers)
                      
                         : "t0", "t1", "f0", "f1", "f30", "f31", "memory");
    
    auto scale = k + normedAlpha * squareSum;

    tScale[offsetOut] = scale;

    auto normFactor = getPow(scale, -beta);
    auto op = tAInput[offsetIn];
    op *= normFactor;
    tOutput[offsetOut] = op;
    


    srcAddr = (uintptr_t)activations;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}





//===----------------------------------------------------------------------===//
//                       Arithmetic operations
//===----------------------------------------------------------------------===//
//

inline __attribute__((always_inline)) void
getCoordinates(unsigned int iDx, unsigned int srcDimNum, unsigned int *sizes,
               unsigned int *coordinates) {

  unsigned int acum = sizes[srcDimNum - 1];
  coordinates[srcDimNum - 1] = iDx % acum;
  // it assumes sizes is never 0 fot the srcDimNum dimensions
  for (int dimI = srcDimNum - 2; dimI >= 0; dimI--) {
    if (iDx >= acum) {
      coordinates[dimI] = (iDx / acum) % (sizes[dimI]);
      acum *= sizes[dimI];
    } else {
      break;
    }
  }
}

inline __attribute__((always_inline)) uint64_t
calcAddrOffset(unsigned int *pitches, unsigned int *coordinates) {
  return (pitches[0] * coordinates[0]) + (pitches[1] * coordinates[1]) +
         (pitches[2] * coordinates[2]) + (pitches[3] * coordinates[3]) +
         (pitches[4] * coordinates[4]) + (pitches[5] * coordinates[5]);
}

template <typename srcType, typename opType>
void dnn_lib::fwdLibElementInst(void *dstT, void *dstDims, void *dstPitches,
                                void *srcT1, void *srcDims, void *src1Pitches,
                                unsigned int srcDimNum, void *srcT2, 
                                void *src2Pitches, float *scale, 
                                int32_t *offset) {
 
 unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> aSrcT2(srcT2, scale[1], offset[1]);
  Addresser<srcType> aDstT(dstT, scale[2], offset[2]);

  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *act1Pitch = (unsigned int *)src1Pitches;
  unsigned int *act2Pitch = (unsigned int *)src2Pitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc1Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc2Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrc1Pitch[i] = act1Pitch[i];
    eSrc2Pitch[i] = act2Pitch[i];
  }

  uint64_t addrSrc1, addrSrc2, addrDst;

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, opType> op;

  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc1 = x * eSrc1Pitch[0] + y * eSrc1Pitch[1] + z * eSrc1Pitch[2] +
                        w * eSrc1Pitch[3] + q * eSrc1Pitch[4] + r * eSrc1Pitch[5];
              addrSrc2 = x * eSrc2Pitch[0] + y * eSrc2Pitch[1] + z * eSrc2Pitch[2] +
                        w * eSrc2Pitch[3] + q * eSrc2Pitch[4] + r * eSrc2Pitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              op.doOp(aDstT, aSrcT1, aSrcT2, addrDst, addrSrc1, addrSrc2);
            }
          }
        }
      }
    }
  }
}



template <typename srcType, typename opType>
void dnn_lib::fwdLibElementInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *src1Pitches, unsigned int srcDimNum, void *srcT2, void *src2Pitches, float *scale,
    int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> aSrcT2(srcT2, scale[1], offset[1]);
  Addresser<srcType> aDstT(dstT, scale[2], offset[2]);

  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *act1Pitch = (unsigned int *)src1Pitches;
  unsigned int *act2Pitch = (unsigned int *)src2Pitches;

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn1 = 0;
  uint64_t offsetIn2 = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn1 += act1Pitch[j] * coord[j];
    offsetIn2 += act2Pitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    op.doOp(aDstT, aSrcT1, aSrcT2, offsetOut, offsetIn1, offsetIn2);
    done = getOffsets(srcDimNum, coord, offsetIn1, offsetIn2, offsetOut, actIndex,
                      act1Pitch, act2Pitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}



/* This is a very similar implementation to CopyInst, it is recommended to
 * read it in order to more easily understand it, as it is well commented
 * there.*/

template <typename src1Type, typename src2Type, typename dstType, typename opType>
void dnn_lib::fwdLibElementInstVectorized(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *src1Pitches, unsigned int srcDimNum, void *srcT2, void *src2Pitches,
    float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)src1Pitches;
  uintptr_t dstAddr = (uintptr_t)dstT;
  uintptr_t srcAddr1 = (uintptr_t)srcT1;
  uintptr_t srcAddr2 = (uintptr_t)srcT2;

  Operator<Addresser<src1Type>, Addresser<src2Type>, Addresser<dstType>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<src1Type>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;

  volatile int32_t gatherValues[] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (unsigned int i = 0; i < 8; i++) {
      gatherValues[i] = i * typeSize;
  }

  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  unsigned int elementsInRow, registersInRow, res;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

  while (!done && (offsetOut < posMax)) {
    if (firstRow && coord[lastDim - 1] != maxRow) {
      elementsInRow = actIndex[lastDim] - coord[lastDim];
    } else if (coord[lastDim - 1] == maxRow) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
    } else {
      elementsInRow = actIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      res = elementsInRow - registersInRow * 8;
      mask = ((1 << res) - 1);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr1 += offsetIn * typeSize; 
    srcAddr2 += offsetIn * typeSize; 
    dstAddr += offsetOut * typeSize;
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

    unsigned int cnt = 0;
    
    while(cnt < registersInRow) {
      op.doOpVect(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset); 
      cnt++;
      srcAddr1 += 8 * typeSize;
      srcAddr2 += 8 * typeSize;
      dstAddr += 8 * typeSize;
    }
    if (res > 0) {
      __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      op.doOpVect(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset); 
    } 

    if (lastRow) 
      return;
    
    dstAddr = (uintptr_t)dstT;
    srcAddr1 = (uintptr_t)srcT1;
    srcAddr2 = (uintptr_t)srcT2;
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;
    done = getOffsets(lastDim , coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}


template <typename srcType, typename opType>
void dnn_lib::fwdLibElementBoolInst(void *dstT, void *dstDims, void *dstPitches,
                                    void *srcT1, void *srcDims,
                                    void *src1Pitches, unsigned int srcDimNum,
                                    void *srcT2, void *src2Pitches, float *scale,
                                    int32_t *offset) {

  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> aSrcT2(srcT2, scale[1], offset[1]);
  bool *aDstT = (bool *)dstT;

  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *src1Pitch = (unsigned int *)src1Pitches;
  unsigned int *src2Pitch = (unsigned int *)src2Pitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc1Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc2Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrc1Pitch[i] = src1Pitch[i];
    eSrc2Pitch[i] = src2Pitch[i];
  }

  uint64_t addrSrc1, addrSrc2, addrDst;

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, opType> op;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc1 = x * eSrc1Pitch[0] + y * eSrc1Pitch[1] +
                         z * eSrc1Pitch[2] + w * eSrc1Pitch[3] +
                         q * eSrc1Pitch[4] + r * eSrc1Pitch[5];
              addrSrc2 = x * eSrc2Pitch[0] + y * eSrc2Pitch[1] +
                         z * eSrc2Pitch[2] + w * eSrc2Pitch[3] +
                         q * eSrc2Pitch[4] + r * eSrc2Pitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              op.doOp(aDstT, aSrcT1, aSrcT2, addrDst, addrSrc1, addrSrc2);
            }
          }
        }
      }
    }
  }
}

/* This is a very similar implementation to CopyInst, it is recommended to
 * readed in order to more easily understand it, as it is well commented
 * there.*/
template <typename srcType, typename opType>
void dnn_lib::fwdLibElementBoolInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *src1Pitches, unsigned int srcDimNum, void *srcT2, void *src2Pitches,
    float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> aSrcT2(srcT2, scale[1], offset[1]);
  bool *aDstT = (bool *)dstT;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *act1Pitch = (unsigned int *)src1Pitches;
  unsigned int *act2Pitch = (unsigned int *)src2Pitches;
  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn1 = 0;
  uint64_t offsetIn2 = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn1 += act1Pitch[j] * coord[j];
    offsetIn2 += act2Pitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    op.doOp(aDstT, aSrcT1, aSrcT2, offsetOut, offsetIn1, offsetIn2);
    done = getOffsets(srcDimNum, coord, offsetIn1, offsetIn2, offsetOut, actIndex,
                      act1Pitch, act2Pitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}


template <typename src1Type, typename src2Type, typename opType>
void dnn_lib::fwdLibElementBoolInstVectorized(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *src1Pitches, unsigned int srcDimNum, void *srcT2, void *src2Pitches,
    float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)src1Pitches;
  bool *dstAddr = (bool *)dstT;
  uintptr_t srcAddr1 = (uintptr_t)srcT1;
  uintptr_t srcAddr2 = (uintptr_t)srcT2;

  Operator<Addresser<src1Type>, Addresser<src2Type>, Addresser<src2Type>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<src1Type>();
  getCachelinePartition(1, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;

  volatile int32_t gatherValues[] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (unsigned int i = 0; i < 8; i++) {
      gatherValues[i] = i * typeSize;
  }
  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  unsigned int elementsInRow, registersInRow, res;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);
 
  while (!done && (offsetOut < posMax)) {
    if (firstRow && coord[lastDim - 1] != maxRow) {
      elementsInRow = actIndex[lastDim] - coord[lastDim];
    } else if (coord[lastDim - 1] == maxRow) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
    } else {
      elementsInRow = actIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      res = elementsInRow - registersInRow * 8;
      mask = ((1 << res) - 1);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr1 += offsetIn * typeSize; 
    srcAddr2 += offsetIn * typeSize; 
    dstAddr += offsetOut;

    unsigned int cnt = 0;
    while(cnt < registersInRow) {
      __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
      op.doOpVect(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset); 
      cnt++;
      srcAddr1 += 8 * typeSize;
      srcAddr2 += 8 * typeSize;
      dstAddr += 8;
    }
    if (res > 0) {
      __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      op.doOpVect(gatherValues, srcAddr1, srcAddr2, dstAddr, scale, offset); 
    }
    if (lastRow) 
      return;
    
    dstAddr = (bool *)dstT;
    srcAddr1 = (uintptr_t)srcT1;
    srcAddr2 = (uintptr_t)srcT2;
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;
    done = getOffsets(lastDim , coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename srcType, typename opType>
void dnn_lib::fwdLibElementSingleInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *srcT1,
                                      void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, float *scale,
                                      int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  Addresser<srcType> aDstT(dstT, scale[2], offset[2]);

  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc1, addrSrc2, addrDst;

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, opType> op;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc1 = x * eSrcPitch[0] + y * eSrcPitch[1] +
                         z * eSrcPitch[2] + w * eSrcPitch[3] +
                         q * eSrcPitch[4] + r * eSrcPitch[5];
              addrSrc2 = x * eSrcPitch[0] + y * eSrcPitch[1] +
                         z * eSrcPitch[2] + w * eSrcPitch[3] +
                         q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              op.doOp(aDstT, aSrcT1, addrDst, addrSrc1);
            }
          }
        }
      }
    }
  }
}

template <typename srcType, typename opType>
void dnn_lib::fwdLibElementSingleInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, float *scale,
    int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  Addresser<srcType> aDstT(dstT, scale[2], offset[2]);

  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    op.doOp(aDstT, aSrcT1, offsetOut, offsetIn);
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename src1Type, typename dstType, typename opType>
void dnn_lib::fwdLibElementSingleInstVectorized(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, float *scale,
    int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;
  uintptr_t dstAddr = (uintptr_t)dstT;
  uintptr_t srcAddr = (uintptr_t)srcT1;

  Operator<Addresser<src1Type>, Addresser<src1Type>, Addresser<dstType>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<src1Type>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;

  volatile int32_t gatherValues[] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (unsigned int i = 0; i < 8; i++) {
      gatherValues[i] = i * typeSize;
  }

  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  unsigned int elementsInRow, registersInRow, res, spareElems, fullLanes;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false; 
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);
 
  while (!done && (offsetOut < posMax)) {
    if (firstRow && coord[lastDim - 1] != maxRow) {
      elementsInRow = actIndex[lastDim] - coord[lastDim];
    } else if (coord[lastDim - 1] == maxRow) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
    } else {
      elementsInRow = actIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      res = elementsInRow - registersInRow * 8;
      mask = ((1 << res) - 1);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr += offsetIn * typeSize; 
    dstAddr += offsetOut * typeSize;
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

    unsigned int cnt = 0;
    while(cnt < registersInRow) {
      op.doOpVect(gatherValues, srcAddr, dstAddr, scale, offset); 
      cnt++;
      srcAddr += 8 * typeSize;
      dstAddr += 8 * typeSize;
    }
    if (res > 0) {
      __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      op.doOpVect(gatherValues, srcAddr, dstAddr, scale, offset); 
    }
    if (lastRow) 
      return;
    
    dstAddr = (uintptr_t)dstT;
    srcAddr = (uintptr_t)srcT1;
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;
    done = getOffsets(lastDim , coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibElementIsNaNInst(void *dstT, void *dstDims,
                                     void *dstPitches, void *srcT1,
                                     void *srcDims, void *srcPitches,
                                     unsigned int srcDimNum, float *scale,
                                     int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  bool *ptrDstT = (bool *)dstT;
  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              ptrDstT[x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                      w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5]] =
                  isnanf(aSrcT1[x * eSrcPitch[0] + y * eSrcPitch[1] +
                                z * eSrcPitch[2] + w * eSrcPitch[3] +
                                q * eSrcPitch[4] + r * eSrcPitch[5]])
                      ? true
                      : false;
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibElementIsNaNInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT1, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, float *scale, int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> aSrcT1(srcT1, scale[0], offset[0]);
  bool *aDstT = (bool *)dstT;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;
  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    aDstT[offsetOut] = isnanf(aSrcT1[offsetIn]) ? true : false;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibElementSelectInst(
    void *dstT, void *dstDims, void *dstPitches, void *condT, void *condDims,
    void *condPitches, void *srcT1, void *srcDims, void *src1Pitches,
    unsigned int srcDimNum, void *srcT2, void *src2Pitches, 
    float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> ptrDstT(dstT, scale[3], offset[3]);
  bool *ptrCondT = (bool *)condT;
  const Addresser<srcType> ptrSrcT1(srcT1, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT2(srcT2, scale[2], offset[2]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;
  unsigned int *condIndex = (unsigned int *)condDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *src1Pitch = (unsigned int *)src1Pitches;
  unsigned int *src2Pitch = (unsigned int *)src2Pitches;
  unsigned int *condPitch = (unsigned int *)condPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc1Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc2Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eCondPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrc1Pitch[i] = src1Pitch[i];
    eSrc2Pitch[i] = src2Pitch[i];
    eCondPitch[i] = condPitch[i];
  }

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              size_t src1I = x * eSrc1Pitch[0] + y * eSrc1Pitch[1] +
                             z * eSrc1Pitch[2] + w * eSrc1Pitch[3] +
                             q * eSrc1Pitch[4] + r * eSrc1Pitch[5];
              size_t src2I = x * eSrc2Pitch[0] + y * eSrc2Pitch[1] +
                             z * eSrc2Pitch[2] + w * eSrc2Pitch[3] +
                             q * eSrc2Pitch[4] + r * eSrc2Pitch[5];
              size_t dstI = x * eDstPitch[0] + y * eDstPitch[1] +
                            z * eDstPitch[2] + w * eDstPitch[3] +
                            q * eDstPitch[4] + r * eDstPitch[5];
              size_t condI = x * eCondPitch[0] + y * eCondPitch[1] +
                             z * eCondPitch[2] + w * eCondPitch[3] +
                             q * eCondPitch[4] + r * eCondPitch[5];
              ptrDstT[dstI] =
                  (ptrCondT[condI]) ? ptrSrcT1[src1I] : ptrSrcT2[src2I];
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibElementSelectInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *condT, void *condDims,
    void *condPitches, void *srcT1, void *srcDims, void *src1Pitches,
    unsigned int srcDimNum, void *srcT2, void *src2Pitches, float *scale, int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> ptrDstT(dstT, scale[3], offset[3]);
  bool *ptrCondT = (bool *)condT;
  const Addresser<srcType> ptrSrcT1(srcT1, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT2(srcT2, scale[2], offset[2]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;
  unsigned int *condIndex = (unsigned int *)condDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *act1Pitch = (unsigned int *)src1Pitches;
  unsigned int *act2Pitch = (unsigned int *)src2Pitches;
  unsigned int *condPitch = (unsigned int *)condPitches;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  unsigned int offsetIn1 = 0;
  unsigned int offsetIn2 = 0;
  unsigned int offsetOut = 0;
  unsigned int offsetCond = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn1 += act1Pitch[j] * coord[j];
    offsetIn2 += act2Pitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
    offsetCond += condPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    ptrDstT[offsetOut] =
        (ptrCondT[offsetCond]) ? ptrSrcT1[offsetIn1] : ptrSrcT2[offsetIn2];
    done = getOffsets(srcDimNum, coord, offsetIn1, offsetIn2, offsetOut, 
       offsetCond, actIndex, act1Pitch, act2Pitch, dstPitch, condPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibModuloInst(void *dstT, void *dstDims, void *dstPitches,
                               void *srcT, void *srcDims, void *srcPitches,
                               unsigned int srcDimNum, long long divisor,
                               bool signFollowDivisor, float *scale,
                               int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  srcType *tOutput = (srcType *)dstT;
  srcType *tInput = (srcType *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;

  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              auto res = (tInput[addrSrc]) % divisor;
              if (signFollowDivisor && (res < 0)) {
                res += divisor;
              }
              tOutput[addrDst] = res;
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibModuloInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, long long divisor,
    bool signFollowDivisor, float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);
  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    auto res = (tInput[offsetIn]) % divisor;
    if (signFollowDivisor && (res < 0)) {
      res += divisor;
    }
    tOutput[offsetOut] = res;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibElementExpInst(void *dstT, void *dstDims, void *dstPitches,
                                   void *srcT, void *srcDims, void *srcPitches,
                                   unsigned int srcDimNum, float *scale,
                                   int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  srcType *tOutput = (srcType *)dstT;
  srcType *tInput = (srcType *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;

  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              auto val = tInput[addrSrc];
	      float res = getExp((float)val);
              tOutput[addrDst] = res;
            }
          }
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
//                       Mat Mul
//===----------------------------------------------------------------------===//
template <typename srcType>
void dnn_lib::fwdLibMatMulInst(void *dstMatrix, void *dstMatrixDims,
                               void *dstMatrixPitches, void *activations,
                               void *activationsDims, void *activationsPitches,
                               void *weights, void *weightsDims,
                               void *weightPitches, float *scale,
                               int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[2], offset[2]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  // For each (x,y) in the destination matrix:
  for (unsigned int x = 0; x < dstIndex[0]; x++) {
    for (unsigned int y = 0; y < dstIndex[1]; y++) {
      // Perform DOT on the row an column.
      float sum = 0;
      for (unsigned int i = 0; i < actIndex[1]; i++) {
        sum += float(tAInput[x * actPitch[0] + i]) *
               float(tWInput[i * weightPitch[0] + y]);
      }
      tOutput[x * dstPitch[0] + y] = float(sum);
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibMatMulInstTransposed(void *dstMatrix, void *dstMatrixDims,
                                         void *dstMatrixPitches, void *activations,
                                         void *activationsDims, void *activationsPitches,
                                         void *weights, void *weightsDims,
                                         void *weightPitches, float *scale,
                                         int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[2], offset[2]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  // For each (x,y) in the destination matrix:
  for (unsigned int x = 0; x < dstIndex[0]; x++) {
    for (unsigned int y = 0; y < dstIndex[1]; y++) {
      // Perform DOT on the row an column.
      float sum = 0;
      for (unsigned int i = 0; i < actIndex[1]; i++) {
        sum += float(tAInput[x * actPitch[0] + i]) *
               float(tWInput[y * weightPitch[0] + i]);
      }
      tOutput[x * dstPitch[0] + y] = float(sum);
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibMatMulInstThreaded(void *dstMatrix, void *dstMatrixDims,
                                       void *dstMatrixPitches,
                                       void *activations, void *activationsDims,
                                       void *activationsPitches, void *weights,
                                       void *weightsDims, void *weightPitches,
                                       float *scale, int32_t *offset,
                                       uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[2], offset[2]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int k = 0;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coordOut[i] * dstPitch[i];
  }
  unsigned int offsetAIn = coordOut[0]*actPitch[0];

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    float sum = 0;
    unsigned int weightOffset = 0;
    for (unsigned int i = 0; i < actIndex[1]; i++) {
      sum += float(tAInput[offsetAIn + i]) *
             float(tWInput[weightOffset + coordOut[1]]);
      weightOffset += weightPitch[0];
    }
    tOutput[offsetOut] = float(sum);
    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] == 0) {
      offsetAIn += actPitch[0];
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibMatMulInstThreadedTransposed(void *dstMatrix, void *dstMatrixDims,
                                                 void *dstMatrixPitches,
                                                 void *activations, void *activationsDims,
                                                 void *activationsPitches, void *weights,
                                                 void *weightsDims, void *weightPitches,
                                                 float *scale, int32_t *offset,
                                                 uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[2], offset[2]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int k = 0;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coordOut[i] * dstPitch[i];
  }

  uint64_t offsetAIn = coordOut[0]*actPitch[0];
  uint64_t offsetWIn = coordOut[1]*weightPitch[0];

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    float sum = 0;
    for (unsigned int i = 0; i < actIndex[1]; i++) {
      sum += float(tAInput[offsetAIn + i]) * float(tWInput[offsetWIn + i]);
    }
    tOutput[offsetOut] = float(sum);
    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] != 0) {
      offsetWIn += weightPitch[0];
    }
    else {
      offsetWIn = 0;
      offsetAIn += actPitch[0];
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, float *scale, int32_t *offset){

#define MATMUL_ITERATION               \
    "flw.ps   f0, 0x0(%[actAddr])\n"   \
    "fgw.ps   f1, f29(%[wgtAddr])\n"   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x20\n"
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"
                                         
    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fsw.ps f31, 0x0(%[dstAddr])\n"
    
    :
    : [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr)
    : "t0", "t1", "f0", "f1", "f29", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, float *scale, int32_t *offset){

#define MATMUL_ITERATION               \
    "fgh.ps   f0, f28(%[actAddr])\n"   \
    "fgh.ps   f1, f29(%[wgtAddr])\n"   \
    "fcvt.ps.f16 f0, f0 \n"            \
    "fcvt.ps.f16 f1, f1 \n"            \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f28, 0x0(%[gthValuesAct])\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x10\n" 
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"
                                         
    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "fcvt.f16.ps f31, f31\n"           // Conversion fp32 >> fp16.
    "mov.m.x m0, zero, 0x1\n"
    "fsch.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr)
    : "t0", "t1", "f0", "f1", "f28", "f29", "f30", "f31", "memory");

#undef MATMUL_ITERATION

}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, float *scale, int32_t *offset){

#define INT8_TO_FP32(_reg)                  \
    "fsub.pi " #_reg ", " #_reg ", f26 \n"  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"    \
    "fmul.ps " #_reg ", " #_reg ", f27 \n"

#define MATMUL_ITERATION               \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f26, 0x0(%[offset]) \n"  \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                   \
    "fbc.ps   f26, 0x4(%[offset]) \n"  \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define FP32_TO_INT8(_reg)                      \
    "frcp.ps f27, f27 \n"                       \
    "fcvt.ps.pw f26, f26 \n"                    \
    "fmadd.ps " #_reg ", " #_reg ", f27, f26 \n" \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsat8.pi " #_reg ", " #_reg "\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f28, 0x0(%[gthValuesAct])\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x8\n"
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"
                                         
    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f26, 0x8(%[offset]) \n"
    "fbc.ps f27, 0x8(%[scale]) \n"
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

#undef INT8_TO_FP32
#undef MATMUL_ITERATION
#undef FP32_TO_INT8

}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value && !std::is_same<srcType, float16>::value && !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void matmulOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, float *scale, int32_t *offset){}

// Default version of MatMul in case the input weights have not been previously transposed.
template <typename srcType>
void dnn_lib::fwdLibMatMulInstVectorized(void *dstMatrix, void *dstMatrixDims,
                                         void *dstMatrixPitches,
                                         void *activations, void *activationsDims,
                                         void *activationsPitches, void *weights,
                                         void *weightsDims, void *weightPitches,
                                         float *scale, int32_t *offset,
                                         uint64_t flags,
                                         const uint32_t minionOffset,
                                         const uint32_t assignedMinions) {

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (32 * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[2], offset[2]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int last_non_zero_coord = 0;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, last_non_zero_coord);

  unsigned int offsetOut = 0;
  for (int i = 0; i < last_non_zero_coord; i++) {
    offsetOut += coordOut[i] * dstPitch[i];
  }
  unsigned int offsetAIn = coordOut[0]*actPitch[0];

  int32_t gatherValuesAct[8], gatherValuesWgt[8];
  gatherValuesAct[0] = gatherValuesWgt[0] = 0;
  unsigned int step = weightPitch[0]*typeSize;
  for (unsigned int i = 1; i < 8; ++i) {
    gatherValuesAct[i] = gatherValuesAct[i - 1] + typeSize;
    gatherValuesWgt[i] = gatherValuesWgt[i - 1] + step;
  }
  unsigned int wgtRegStep = 8*step;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    uintptr_t dstAddr = (uintptr_t)dstMatrix + typeSize*offsetOut;
    uintptr_t actAddr = (uintptr_t)activations + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)weights + typeSize*coordOut[1];
    matmulOp <srcType>(dstAddr, actAddr, wgtAddr, actIndex[1], gatherValuesAct,
                       gatherValuesWgt, wgtRegStep, scale, offset);
    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] == 0) {
      offsetAIn += actPitch[0];
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value, std::size_t>::type = 0>
void matmulOpTrans(uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValues[], float *scale, int32_t *offset){

#define MATMUL_ITERATION               \
    "flw.ps   f0, 0x0(%[actAddr])\n"   \
    "flw.ps   f1, 0x0(%[wgtAddr])\n"   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"        // Mask m0 is set so all lanes are active.
    "xor t0, t0, t0\n"                // The int register t0 is set to 0x0: it will count iterations.
    "fxor.pi f31, f31, f31\n"         // Vectorial register f31 set to 0x0. Only useful lanes: e0, e4.

    "1:\n"                            // New loop (tag 1): vectorised scalar product.
    "addi     t0, t0, 8\n"              // t0 += 8.
    "ble      %[elemsRow], t0, 2f\n"    // if (elemsRow <= t0), forward to tag 2.
    MATMUL_ITERATION                    // The scalar product of the act and weights is added to f31.
    "addi %[actAddr], %[actAddr], 0x20\n"     
    "addi %[wgtAddr], %[wgtAddr], 0x20\n"
    "beq      zero, zero, 1b\n"       // Go back to tag 1.
                                         
    "2:\n"                            // Tag 2: a new mask is set to finish the row's product.
    "fxor.pi  f0, f0, f0\n"           // f0 is set to 0's to get a correct final matmul iteration.
    "addi     t0, t0, -8\n"           // In these two instructions,
    "sub      t0, %[elemsRow], t0\n"  // we update t0 = elemsRow - (t0 - 8).
    "addi     t1, zero, 1\n"          // t1 is set to 1.
    "sll      t1, t1, t0\n"           // Shift Left Logical t0 positions: t1 = 2^(t0).
    "addi     t1, t1, -1\n"           // Finally, t1 = 2^(t0) - 1.
    "mov.m.x  m0, t1, 0\n"            // The mask is set to t1, so the first t0 lanes are active.
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"        // Finally, %[sum] = f31.e0 + f31.e4.
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fsw.ps f31, 0x0(%[dstAddr])\n"
    
    :
    : [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr)
    : "t0", "t1", "f0", "f1", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, std::size_t>::type = 0>
void matmulOpTrans(uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValues[], float *scale, int32_t *offset){

#define MATMUL_ITERATION               \
    "fgh.ps   f0, f30(%[actAddr])\n"   \
    "fgh.ps   f1, f30(%[wgtAddr])\n"   \
    "fcvt.ps.f16 f0, f0 \n"            \
    "fcvt.ps.f16 f1, f1 \n"            \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"        // Mask m0 is set so all lanes are active.
    "xor t0, t0, t0\n"                // The int register t0 is set to 0x0: it will count iterations.
    "flw.ps f30, 0x0(%[gthValues])\n" // The gatherValues vector is loaded to f30, one int32 per lane.
    "fxor.pi f31, f31, f31\n"         // Vectorial register f31 set to 0x0. Only useful lanes: e0, e4.

    "1:\n"                            // New loop (tag 1): vectorised scalar product.
    "addi     t0, t0, 8\n"              // t0 += 8.
    "ble      %[elemsRow], t0, 2f\n"    // if (elemsRow <= t0), forward to tag 2.
    MATMUL_ITERATION                    // The scalar product of the act and weights is added to f31.
    "faddi.pi f30, f30, 0x10\n"         // The gather offset values are updated adding 8 positions.
    "beq      zero, zero, 1b\n"       // Go back to tag 1.
                                         
    "2:\n"                            // Tag 2: a new mask is set to finish the row's product.
    "fxor.pi  f0, f0, f0\n"           // f0 is set to 0's to get a correct final matmul iteration.
    "addi     t0, t0, -8\n"           // In these two instructions,
    "sub      t0, %[elemsRow], t0\n"  // we update t0 = elemsRow - (t0 - 8).
    "addi     t1, zero, 1\n"          // t1 is set to 1.
    "sll      t1, t1, t0\n"           // Shift Left Logical t0 positions: t1 = 2^(t0).
    "addi     t1, t1, -1\n"           // Finally, t1 = 2^(t0) - 1.
    "mov.m.x  m0, t1, 0\n"            // The mask is set to t1, so the first t0 lanes are active.
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"        // Finally, %[sum] = f31.e0 + f31.e4.
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "fcvt.f16.ps f31, f31\n"           // Conversion fp32 >> fp16.
    "mov.m.x m0, zero, 0x1\n"
    "fsc32h.ps f31, zero(%[dstAddr])\n"
    
    :
    : [gthValues] "r" (gatherValues),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr)
    : "t0", "t1", "f0", "f1", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value, std::size_t>::type = 0>
void matmulOpTrans(uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValues[], float *scale, int32_t *offset){

#define INT8_TO_FP32(_reg)                  \
    "fsub.pi " #_reg ", " #_reg ", f28 \n"  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"    \
    "fmul.ps " #_reg ", " #_reg ", f29 \n"

#define MATMUL_ITERATION             \
    "fgb.ps   f0, f30(%[actAddr])\n" \
    "fgb.ps   f1, f30(%[wgtAddr])\n" \
    "fbc.ps f28, 0x0(%[offset]) \n"  \
    "fbc.ps f29, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                 \
    "fbc.ps f28, 0x4(%[offset]) \n"  \
    "fbc.ps f29, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                 \
    "fmul.ps    f0, f0, f1\n"        \
    "fswizz.ps  f1, f0, 0xe\n"       \
    "fadd.ps    f0, f0, f1\n"        \
    "fswizz.ps  f1, f0, 0x1\n"       \
    "fadd.ps    f0, f0, f1\n"        \
    "fadd.ps    f31, f0, f31\n"

#define FP32_TO_INT8(_reg)                      \
    "frcp.ps f29, f29 \n"                       \
    "fcvt.ps.pw f28, f28 \n"                    \
    "fmadd.ps " #_reg ", " #_reg ", f29, f28 \n" \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsat8.pi " #_reg ", " #_reg "\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"        // Mask m0 is set so all lanes are active.
    "xor t0, t0, t0\n"                // The int register t0 is set to 0x0: it will count iterations.
    "flw.ps f30, 0x0(%[gthValues])\n" // The gatherValues vector is loaded to f30, one int32 per lane.
    "fxor.pi f31, f31, f31\n"         // Vectorial register f31 set to 0x0. Only useful lanes: e0, e4.

    "1:\n"                            // New loop (tag 1): vectorised scalar product.
    "addi     t0, t0, 8\n"              // t0 += 8.
    "ble      %[elemsRow], t0, 2f\n"    // if (elemsRow <= t0), forward to tag 2.
    MATMUL_ITERATION                    // The scalar product of the act and weights is added to f31.
    "faddi.pi f30, f30, 0x8\n"          // The gather offset values are updated adding 8 positions.
    "beq      zero, zero, 1b\n"       // Go back to tag 1.
                                         
    "2:\n"                            // Tag 2: a new mask is set to finish the row's product.
    "fxor.pi  f0, f0, f0\n"           // f0 is set to 0's to get a correct final matmul iteration.
    "addi     t0, t0, -8\n"           // In these two instructions,
    "sub      t0, %[elemsRow], t0\n"  // we update t0 = elemsRow - (t0 - 8).
    "addi     t1, zero, 1\n"          // t1 is set to 1.
    "sll      t1, t1, t0\n"           // Shift Left Logical t0 positions: t1 = 2^(t0).
    "addi     t1, t1, -1\n"           // Finally, t1 = 2^(t0) - 1.
    "mov.m.x  m0, t1, 0\n"            // The mask is set to t1, so the first t0 lanes are active.
    MATMUL_ITERATION

    "fmvs.x.ps t0, f31, 0x4\n"        // Finally, sum = f31.e0 + f31.e4.
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"       // Now, the sum is in f31.e0.
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f28, 0x8(%[offset]) \n"
    "fbc.ps f29, 0x8(%[scale]) \n"
    FP32_TO_INT8(f31)
    "fsc32b.ps f31, zero(%[dstAddr])\n"
    
    :
    : [gthValues] "r" (gatherValues),
      [elemsRow] "r" (elemsRow),
      [offset] "r" (offset),
      [scale] "r" (scale),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr)
    : "t0", "t1", "f0", "f1", "f28", "f29", "f30", "f31", "memory");

#undef INT8_TO_FP32
#undef MATMUL_ITERATION
#undef FP32_TO_INT8

}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value && !std::is_same<srcType, float16>::value && !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void matmulOpTrans (uintptr_t dstAddr, intptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValues[], float *scale, int32_t *offset){}

// Version assuming the weights tensor is transposed. Used for CONSTANT tensors
template <typename srcType>
void dnn_lib::fwdLibMatMulInstVectorizedTransposed(void *dstMatrix, void *dstMatrixDims,
                                                   void *dstMatrixPitches,
                                                   void *activations, void *activationsDims,
                                                   void *activationsPitches, void *weights,
                                                   void *weightsDims, void *weightPitches,
                                                   float *scale, int32_t *offset,
                                                   uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int last_non_zero_coord;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, last_non_zero_coord);

  uint64_t offsetOut = 0;
  for (unsigned int i = 0; i < last_non_zero_coord; i++) {
    offsetOut += dstPitch[i]*coordOut[i];
  }

  uint64_t offsetAIn = coordOut[0]*actPitch[0];
  uint64_t offsetWIn = coordOut[1]*weightPitch[0];

  int32_t gatherValues[8];
  if (typeSize < 4) {
    gatherValues[0] = 0;
    for (unsigned int i = 1; i < 8; ++i)
      gatherValues[i] = gatherValues[i - 1] + typeSize;
  }

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    uintptr_t dstAddr = (uintptr_t)dstMatrix + typeSize*offsetOut;
    uintptr_t actAddr = (uintptr_t)activations + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)weights + typeSize*offsetWIn;
    matmulOpTrans <srcType>(dstAddr, actAddr, wgtAddr, actIndex[1], gatherValues, scale, offset);
    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] != 0) {
      offsetWIn += weightPitch[0];
    }
    else {
      offsetWIn = 0;
      offsetAIn += actPitch[0];
    }
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

//===----------------------------------------------------------------------===//
//                       Row-wise quantized FC
//===----------------------------------------------------------------------===//

void dnn_lib::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTy(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  int8_t *tWInput = (int8_t *)pweights;
  int32_t *tBias = (int32_t *)pbias;
  float *tScale = (float *)pscale;
  int32_t *tOffset = (int32_t *)poffset;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  float inversedDstScale;
  fpReciprocalSingleElement(dstscale, inversedDstScale);
  for (size_t i = 0; i < dataIndex[0]; i++) {
    for (size_t j = 0; j < dstIndex[1]; j++) {
      float matMulScale = tScale[j] * srcscale;
      float inversedMatMulScale;
      fpReciprocalSingleElement(matMulScale, inversedMatMulScale);
      int32_t sum = 0;
      for (size_t k = 0; k < dataIndex[1]; k++) {
        int32_t W = tWInput[j * weightPitch[0] + k];
        int32_t A = tAInput[i * dataPitch[0] + k];
        sum += (W - tOffset[j]) * (A - srcoffset);
      }
      int32_t B = nearbyintf(float(tBias[j] - biasoffset) *
                             (biasscale * inversedMatMulScale));
      sum += B;
      // Scale the result back to the expected destination scale.
      tOutput[i * dstPitch[0] + j] = clip<int32_t, int8_t>(nearbyintf(
          float(sum) * (matMulScale * inversedDstScale) + dstoffset));
    }
  }
}


void dnn_lib::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  int8_t *tWInput = (int8_t *)pweights;
  int32_t *tBias = (int32_t *)pbias;
  float *tScale = (float *)pscale;
  int32_t *tOffset = (int32_t *)poffset;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  float invDstScale;
  fpReciprocalSingleElement(dstscale, invDstScale);

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];  
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<int8_t>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions); 
  if (maxRead == 0)
    return;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int last_non_zero_coord;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, last_non_zero_coord);

  uint64_t offsetOut = 0;
  for (unsigned int i = 0; i < last_non_zero_coord; i++) {
    offsetOut += dstPitch[i]*coordOut[i];
  }

  uint64_t offsetAIn = coordOut[0]*dataPitch[0];
  uint64_t offsetWIn = coordOut[1]*weightPitch[0];

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  while (!done && (offsetOut < posMax)) {

    float matMulScale = tScale[coordOut[1]] * srcscale;
    float invMatMulScale;
    fpReciprocalSingleElement(matMulScale, invMatMulScale);   
    int32_t sum = nearbyintf(float(tBias[coordOut[1]] - biasoffset) * biasscale * invMatMulScale);
    int32_t woffset = tOffset[coordOut[1]];
    for (size_t k = 0; k < dataIndex[1]; k++) {
      int32_t W = tWInput[offsetWIn + k];
      int32_t A = tAInput[offsetAIn + k];
      sum += (W - woffset) * (A - srcoffset);
    }
    tOutput[offsetOut] = clip<int32_t, int8_t>(nearbyintf(
      float(sum) * (matMulScale * invDstScale) + dstoffset));

    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] != 0) {
      offsetWIn += weightPitch[0];
    }
    else {
      offsetWIn = 0;
      offsetAIn += dataPitch[0];
    }

  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);
}

void dnn_lib::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyVectorized(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  int8_t *tWInput = (int8_t *)pweights;
  int32_t *tBias = (int32_t *)pbias;
  float *tScale = (float *)pscale;
  int32_t *tOffset = (int32_t *)poffset;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  float invDstScale;
  fpReciprocalSingleElement(dstscale, invDstScale);

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];  
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<int8_t>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions); 
  if (maxRead == 0)
    return;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int last_non_zero_coord;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, last_non_zero_coord);

  uint64_t offsetOut = 0;
  for (unsigned int i = 0; i < last_non_zero_coord; i++) {
    offsetOut += dstPitch[i]*coordOut[i];
  }

  uint64_t offsetAIn = coordOut[0]*dataPitch[0];
  uint64_t offsetWIn = coordOut[1]*weightPitch[0];

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  while (!done && (offsetOut < posMax)) {

    float matMulScale = tScale[coordOut[1]] * srcscale;
    float invMatMulScale;
    fpReciprocalSingleElement(matMulScale, invMatMulScale);   
    int32_t sum = nearbyintf(float(tBias[coordOut[1]] - biasoffset) * biasscale * invMatMulScale);
    int32_t woffset = tOffset[coordOut[1]];
   
    uintptr_t actAddr = (uintptr_t)tAInput + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)tWInput + typeSize*offsetWIn;

#define MATMUL_ITERATION               \
    "fgb.ps   f0, f28(%[actAddr])\n" \
    "fgb.ps   f1, f28(%[wgtAddr])\n" \
    "fsub.pi    f0, f0, f29\n"         \
    "fsub.pi    f1, f1, f30\n"         \
    "fmul.pi    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.pi    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.pi    f0, f0, f1\n"          \
    "fadd.pi    f31, f0, f31\n"

    volatile int32_t gatherValues[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ __volatile__(
      "mov.m.x m0, zero, 0xff\n"        // Mask m0 is set so all lanes are 
                                        //active.
      "addi t0, %[sum], 0x0\n"          // The value of sum is stored in 
                                        //the integer register t0.
      "xor t1, t1, t1\n"                // The int register t1 is set to 
                                        //0x0: it will count iterations.
      "flw.ps f28, 0x0(%[gthValues])\n" // The gatherValues vector is loaded 
                                        //to f28, one int32 per lane.
      "fbc.ps f29, 0x0(%[srcoffset])\n" // The int32 srcoffset is broadcast 
                                        //to the 8 lanes of f29.
      "fbc.ps f30, 0x0(%[woffset])\n"   // The int32 woffset is broadcast to 
                                        //the 8 lanes of f30.
      "fxor.pi f31, f31, f31\n"         // Vectorial register f31 set to 0x0. 
                                        //Only useful lanes: e0, e4.

      "1:\n"                            // New loop (tag 1): vectorised scalar
                                        //product.
      "addi     t1, t1, 8\n"            // t1 += 8.
      "ble      %[elemsRow], t1, 2f\n"  // if (elemsRow <= t1), forward to 
                                        //tag 2.
      MATMUL_ITERATION                  // The scalar product of the data and
                                        //weights is added to f31.
      "faddi.pi f28, f28, 0x8\n"        // The gather offset values are updated
                                        //adding 8 positions.
      "beq      zero, zero, 1b\n"       // Go back to tag 1.
                                           
      "2:\n"                            // Tag 2: a new mask is set to finish 
                                        //the row's product.
      "fxor.pi  f0, f0, f0\n"           // f0 is set to 0's to get a correct 
                                        //final matmul iteration.
      "addi     t1, t1, -8\n"           // In these two instructions,
      "sub      t1, %[elemsRow], t1\n"  // we update t1 = elemsRow - (t1 - 8).
      "addi     t2, zero, 1\n"          // t2 is set to 1.
      "sll      t2, t2, t1\n"           // Shift Left Logical t1 positions: 
                                        //t2 = 2^(t1).
      "addi     t2, t2, -1\n"           // Finally, t2 = 2^(t1) - 1.
      "mov.m.x  m0, t2, 0\n"            // The mask is set to t2, so the first 
                                        //t1 lanes are active.
      MATMUL_ITERATION
      "fmvs.x.ps t1, f31, 0x0\n"        // The sum stored in f31.e0 is stored 
                                        //in the int register t1.
      "add       t0, t0, t1\n"          // This way, it can be summed to its 
                                        //initial value in t0.
      "fmvs.x.ps t1, f31, 0x4\n"        // The same is done for the sum stored 
                                        //in f31.e4,
      "add       t0, t0,  t1\n"         // so t0 has now the total value of the
                                        //scalar product.
      "addi      %[sum], t0, 0x0\n"     // The value in t0 is then stored in 
                                        //the variable "sum".
      
      : [sum] "+r" (sum)
      : [gthValues] "r" (gatherValues),
        [srcoffset] "r" (&srcoffset),
        [woffset]   "r" (&woffset),
        [actAddr] "r" (actAddr),
        [wgtAddr] "r" (wgtAddr),
        [elemsRow]  "r" (dataIndex[1])
      : "t0", "t1", "t2", "f0", "f1", "f29", "f30", "f31");

    tOutput[offsetOut] = clip<int32_t, int8_t>(nearbyintf(
      float(sum) * (matMulScale * invDstScale) + dstoffset));

    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] != 0) {
      offsetWIn += weightPitch[0];
    }
    else {
      offsetWIn = 0;
      offsetAIn += dataPitch[0];
    }

  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);

#undef MATMUL_ITERATION
}

void dnn_lib::fwdLibRowwiseQuantizedFullyConnectedInstInt8QTyAligned32Bytes(
    void *pdst, void *pdstDims, void *pdstPitches, void *pdata, void *pdataDims,
    void *pdataPitches, void *pscale, void *poffset, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pbias, float srcscale,
    int32_t srcoffset, float dstscale, int32_t dstoffset, float biasscale,
    int32_t biasoffset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  int8_t *tWInput = (int8_t *)pweights;
  int32_t *tBias = (int32_t *)pbias;
  float *tScale = (float *)pscale;
  int32_t *tOffset = (int32_t *)poffset;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  float invDstScale;
  fpReciprocalSingleElement(dstscale, invDstScale);

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];  
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<int8_t>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions); 
  if (maxRead == 0)
    return;

  unsigned int dstDimNum = 2;
  unsigned int coordOut[dstDimNum];
  unsigned int last_non_zero_coord;
  getNonPaddingCoordinates(coordOut, initialAddr, dstDimNum, dstPitch, dstIndex, last_non_zero_coord);

  uint64_t offsetOut = 0;
  for (unsigned int i = 0; i < last_non_zero_coord; i++) {
    offsetOut += dstPitch[i]*coordOut[i];
  }

  uint64_t offsetAIn = coordOut[0]*dataPitch[0];
  uint64_t offsetWIn = coordOut[1]*weightPitch[0];

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  while (!done && (offsetOut < posMax)) {

    float matMulScale = tScale[coordOut[1]] * srcscale;
    float invMatMulScale;
    fpReciprocalSingleElement(matMulScale, invMatMulScale);   
    int32_t sum = nearbyintf(float(tBias[coordOut[1]] - biasoffset) * biasscale * invMatMulScale);
    int32_t woffset = tOffset[coordOut[1]];
   
    uintptr_t actAddr = (uintptr_t)tAInput + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)tWInput + typeSize*offsetWIn;

#define MATMUL_ITERATION               \
    "fg32b.ps   f0, t0(%[actAddr])\n"  \
    "fg32b.ps   f1, t0(%[wgtAddr])\n"  \
    "fsub.pi    f0, f0, f29\n"         \
    "fsub.pi    f1, f1, f30\n"         \
    "fmul.pi    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.pi    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.pi    f0, f0, f1\n"          \
    "fadd.pi    f31, f0, f31\n"

    __asm__ __volatile__(
      "mov.m.x m0, zero, 0xff\n"
      SET_FG32B_VAL(t0)
      "xor t1, t1, t1\n"
      "fbc.ps f29, 0x0(%[srcoffset])\n"
      "fbc.ps f30, 0x0(%[woffset])\n"
      "fxor.pi f31, f31, f31\n"         
      "1:\n"                            
      "addi     t1, t1, 8\n"            
      "ble      %[elemsRow], t1, 2f\n"  
      MATMUL_ITERATION                  
      "addi     %[actAddr], %[actAddr], 0x8\n"     
      "addi     %[wgtAddr], %[wgtAddr], 0x8\n"     
      "j 1b\n"       
      "2:\n"                            
      "fxor.pi  f0, f0, f0\n"           
      "addi     t1, t1, -8\n"           
      "sub      t1, %[elemsRow], t1\n"  
      "addi     t2, zero, 1\n"          
      "sll      t2, t2, t1\n"           
      "addi     t2, t2, -1\n"           
      "mov.m.x  m0, t2, 0\n"             
      MATMUL_ITERATION
      "fmvs.x.ps t1, f31, 0x0\n"
      "addi      t0, zero, 0x0\n"        
      "add       t0, t0, t1\n"          
      "fmvs.x.ps t1, f31, 0x4\n"         
      "add       t0, t0,  t1\n"         
      "add       %[sum], t0, %[sum]\n"     
      
      : [sum] "+r" (sum)
      : [actAddr] "r" (actAddr),
        [wgtAddr] "r" (wgtAddr),
        [srcoffset] "r" (&srcoffset),
        [woffset]   "r" (&woffset),
        [elemsRow]  "r" (dataIndex[1])
      : "t0", "t1", "t2", "f0", "f1", "f29", "f30", "f31");

    tOutput[offsetOut] = clip<int32_t, int8_t>(nearbyintf(
      float(sum) * (matMulScale * invDstScale) + dstoffset));

    done = getOffsets(dstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    if (coordOut[1] != 0) {
      offsetWIn += weightPitch[0];
    }
    else {
      offsetWIn = 0;
      offsetAIn += dataPitch[0];
    }
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);

#undef MATMUL_ITERATION
}

//===----------------------------------------------------------------------===//
//                       Batched operations
//===----------------------------------------------------------------------===//
template <typename srcType>
void dnn_lib::fwdLibBatchedAddInst(void *pdst, void *pdstDims,
                                   void *pdstPitches, void *pbatch,
                                   void *pbatchDims, void *pbatchPitches,
                                   unsigned int pbatchDimNum, void *pslice,
                                   float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tBatch(pbatch, scale[0], offset[0]);
  const Addresser<srcType> tSlice(pslice, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eBatchPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < pbatchDimNum; i++) {
    eBatchDims[i] = batchIndex[i];
    eDstPitch[i] = dstPitch[i];
    eBatchPitch[i] = batchPitch[i];
  }

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Add> op;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              // tOutput[x,y,z,w,q,r] = tBatch[x,y,z,w,q,r] + tSlice[y,z,w,q,r];
              uint64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                 z * eDstPitch[2] + w * eDstPitch[3] +
                                 q * eDstPitch[4] + r * eDstPitch[5];
              uint64_t srcAddr1 = x * eBatchPitch[0] + y * eBatchPitch[1] +
                                  z * eBatchPitch[2] + w * eBatchPitch[3] +
                                  q * eBatchPitch[4] + r * eBatchPitch[5];
              uint64_t srcAddr2 = y * eBatchPitch[1] + z * eBatchPitch[2] +
                                  w * eBatchPitch[3] + q * eBatchPitch[4] +
                                  r * eBatchPitch[5];
              op.doOp(tOutput, tBatch, tSlice, dstAddr, srcAddr1, srcAddr2);
            }
          }
        }
      }
    }
  }
}


template <typename srcType>
void dnn_lib::fwdLibBatchedAddInstThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, void *pbatch,
    void *pbatchDims, void *pbatchPitches, unsigned int pbatchDimNum,
    void *pslice, float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tBatch(pbatch, scale[0], offset[0]);
  const Addresser<srcType> tSlice(pslice, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  unsigned int numElemsDst =
      dstPitch[0] * dstIndex[0]; 

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getUniformCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                               activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[pbatchDimNum]; 
  unsigned int k = 0;                 
  getNonPaddingCoordinates(coord, initialAddr, pbatchDimNum, dstPitch,
                           dstIndex, k);

    
  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += batchPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Add> op;

  while (!done && (offsetOut < posMax)) {
    uint64_t offsetIn2 = offsetIn - coord[0]*batchPitch[0];
    op.doOp(tOutput, tBatch, tSlice, offsetOut, offsetIn, offsetIn2);
    done = getOffsets(pbatchDimNum, coord, offsetOut, offsetIn, dstIndex, dstPitch, batchPitch); 
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);
}

// TODO: Special implementation to support int8_t + int32_t sum, as a quick fix
// we implement this function to support it, the correct way is extend the
// BatchedAdd templatized op and the Operator class in order to support 2
// different templates
void dnn_lib::fwdLibBatchedAddInsti8i32(void *pdst, void *pdstDims,
                                        void *pdstPitches, void *pbatch,
                                        void *pbatchDims, void *pbatchPitches,
                                        unsigned int pbatchDimNum, void *pslice,
                                        void *pslicePitches, float *scale,
                                        int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tBatch = (int8_t *)pbatch;   // scale[0],offset[0]);
  int32_t *tSlice = (int32_t *)pslice; // scale[1]

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;
  unsigned int *slicePitch = (unsigned int *)pslicePitches;

  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eBatchPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSlicePitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < pbatchDimNum; i++) {
    eBatchDims[i] = batchIndex[i];
    eDstPitch[i] = dstPitch[i];
    eBatchPitch[i] = batchPitch[i];
    eSlicePitch[i] = slicePitch[i];
  }
  float invDstScale;
  getReciprocal(scale[2], invDstScale);

  float invLargeScale = (1 << 15);
  float largeScale;
  getReciprocal(invLargeScale, largeScale);
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              // tOutput[x,y,z,w,q,r] = tBatch[x,y,z,w,q,r] + tSlice[y,z,w,q,r];
              uint64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                 z * eDstPitch[2] + w * eDstPitch[3] +
                                 q * eDstPitch[4] + r * eDstPitch[5];
              uint64_t srcAddr1 = x * eBatchPitch[0] + y * eBatchPitch[1] +
                                  z * eBatchPitch[2] + w * eBatchPitch[3] +
                                  q * eBatchPitch[4] + r * eBatchPitch[5];
              uint64_t srcAddr2 = y * eSlicePitch[0] + z * eSlicePitch[1] +
                                  w * eSlicePitch[2] + q * eSlicePitch[3] +
                                  r * eSlicePitch[4];

              int32_t batchVal = tBatch[srcAddr1];
              int32_t sliceVal = tSlice[srcAddr2];

              int32_t B = nearbyintf(float(batchVal - offset[0]) *
                                     (scale[0] * invLargeScale));
              int32_t S = nearbyintf(float(sliceVal - offset[1]) *
                                     (scale[1] * invLargeScale));
              int32_t R = B + S;
              tOutput[dstAddr] = clip<int32_t, int8_t>(nearbyintf(
                  float(R) * (largeScale * invDstScale) + offset[2]));
            }
          }
        }
      }
    }
  }
}


void dnn_lib::fwdLibBatchedAddInsti8i32Threaded(void *pdst, void *pdstDims,
                                                void *pdstPitches, void *pbatch,
                                                void *pbatchDims, void *pbatchPitches,
                                                unsigned int pbatchDimNum, void *pslice,
                                                void *pslicePitches, float *scale,
                                                int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

 int8_t *tOutput = (int8_t *)pdst;
  int8_t *tBatch = (int8_t *)pbatch;   // scale[0],offset[0]);
  int32_t *tSlice = (int32_t *)pslice; // scale[1]

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;
  unsigned int *slicePitch = (unsigned int *)pslicePitches;

  float invDstScale;
  getReciprocal(scale[2], invDstScale);

  float invLargeScale = (1 << 15);
  float largeScale;
  getReciprocal(invLargeScale, largeScale);

  unsigned int numElemsDst =
      dstPitch[0] * dstIndex[0]; 

  unsigned int initialAddr, maxRead;
  getUniformCachelinePartition(sizeof(int8_t), numElemsDst, initialAddr, maxRead,
                               activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[pbatchDimNum]; 
  unsigned int k = 0;                 
  getNonPaddingCoordinates(coord, initialAddr, pbatchDimNum, dstPitch,
                           dstIndex, k);

  uint64_t offsetIn2 = 0;
  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;

  unsigned int eSlicePitch[pbatchDimNum];
  eSlicePitch[0] = 0;
  for(int i = 1; i < pbatchDimNum; i++) eSlicePitch[i] = slicePitch[i - 1];
  
  for (unsigned int j = 0; j < k; j++) {
    offsetIn2 += eSlicePitch[j] * coord[j];
    offsetIn += batchPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j]; 
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;


  while (!done && (offsetOut < posMax)) {
    int32_t batchVal = tBatch[offsetIn];
    int32_t sliceVal = tSlice[offsetIn2];

    int32_t B = nearbyintf(float(batchVal - offset[0]) *
                                     (scale[0] * invLargeScale));
    int32_t S = nearbyintf(float(sliceVal - offset[1]) *
                                     (scale[1] * invLargeScale));
    int32_t R = B + S;
    tOutput[offsetOut] = clip<int32_t, int8_t>(nearbyintf(
    float(R) * (largeScale * invDstScale) + offset[2]));

    done = getOffsets(pbatchDimNum, coord, offsetOut, offsetIn, offsetIn2, dstIndex, dstPitch, batchPitch, eSlicePitch); 
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * sizeof(int8_t) / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + sizeof(int8_t)*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibBatchedReduceAddInst(void *pdst, void *pdstDims,
                                         void *pdstPitches, void *pbatch,
                                         void *pbatchDims, void *pbatchPitches,
                                         unsigned int pbatchDimNum,
                                         unsigned int axis, float *scale,
                                         int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(pdst, scale[1], offset[1]);
  const Addresser<srcType> tBatch(pbatch, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eBatchPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < pbatchDimNum; i++) {
    eBatchDims[i] = batchIndex[i];
    if (i < axis) {
      eDstPitch[i] = dstPitch[i];
    } else if (i > axis) {
      eDstPitch[i] = dstPitch[i - 1];
    }
    eBatchPitch[i] = batchPitch[i];
  }

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Add> op;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              uint64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                 z * eDstPitch[2] + w * eDstPitch[3] +
                                 q * eDstPitch[4] + r * eDstPitch[5];
              uint64_t srcAddr = x * eBatchPitch[0] + y * eBatchPitch[1] +
                                 z * eBatchPitch[2] + w * eBatchPitch[3] +
                                 q * eBatchPitch[4] + r * eBatchPitch[5];
              op.doOp(tOutput, tOutput, tBatch, dstAddr, dstAddr, srcAddr);
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibBatchedReduceAddInstThreaded(void *pdst, void *pdstDims,
                                                 void *pdstPitches, void *pbatch,
                                                 void *pbatchDims, void *pbatchPitches,
                                                 unsigned int pbatchDimNum,
                                                 unsigned int axis, float *scale,
                                                 int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(pdst, scale[1], offset[1]);
  const Addresser<srcType> tBatch(pbatch, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  unsigned int numElemsDst;
  
  numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);

  if (maxRead == 0)
    return;

  unsigned int offsets[pbatchDimNum - 1];

  unsigned int k; 

  unsigned int redBatchPitch[pbatchDimNum - 1];
  for (size_t i = 0; i < pbatchDimNum; i++) {
    if (i < axis) {
      redBatchPitch[i] = batchPitch[i];
      
    } else if (i > axis) {
      redBatchPitch[i - 1] = batchPitch[i];
    }
  }

  getNonPaddingCoordinates(offsets, initialAddr, pbatchDimNum - 1, dstPitch, dstIndex,
                           k);
  uint64_t offsetOut = 0;
  uint64_t offsetIn = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * offsets[j];
    offsetIn += redBatchPitch[j] * offsets[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  Addresser<srcType> *tOutputPtr;
  Addresser<srcType> *tSumPtr;
  bool done = false;
  //int sum = 0;
  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Add> op;
  while (!done && (offsetOut < posMax)) {
    for (size_t i = 0; i < batchIndex[axis]; i++) {
      print(__PRETTY_FUNCTION__);
      Addresser<srcType> tSum = tOutput;
      op.doOp(tOutput, tSum, tBatch, offsetOut, offsetOut, offsetIn);
      offsetIn += batchPitch[axis];
    }
    offsetIn -= batchIndex[axis] * batchPitch[axis];
     
    done = getOffsets(pbatchDimNum - 1, offsets, offsetIn, offsetOut, dstIndex,
                      redBatchPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);
}

void dnn_lib::fwdLibBatchedReduceAddInstInt8(
    void *pdst, void *pdstDims, void *pdstPitches, void *pbatch,
    void *pbatchDims, void *pbatchPitches, unsigned int pbatchDimNum,
    unsigned int axis, float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tBatch = (int8_t *)pbatch;

  float invScale;
  getReciprocal(scale[1], invScale);
  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  // assert(pbatchDimNum <= MAX_TENSOR_DIMENSIONS);

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eBatchPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < pbatchDimNum; i++) {
    eBatchDims[i] = batchIndex[i];
    if (i < axis) {
      eDstPitch[i] = dstPitch[i];
    } else if (i > axis) {
      eDstPitch[i] = dstPitch[i - 1];
    }
    eBatchPitch[i] = batchPitch[i];
  }
#define LOOP_AXIS_CASE(_D0, _D1, _D2, _D3, _D4, _D5_AXIS)                      \
  for (size_t i##_D0 = 0; i##_D0 < eBatchDims[_D0]; i##_D0++)                  \
    for (size_t i##_D1 = 0; i##_D1 < eBatchDims[_D1]; i##_D1++)                \
      for (size_t i##_D2 = 0; i##_D2 < eBatchDims[_D2]; i##_D2++)              \
        for (size_t i##_D3 = 0; i##_D3 < eBatchDims[_D3]; i##_D3++)            \
          for (size_t i##_D4 = 0; i##_D4 < eBatchDims[_D4]; i##_D4++) {        \
            float sum = 0.0;                                                   \
            for (size_t i##_D5_AXIS = 0; i##_D5_AXIS < eBatchDims[_D5_AXIS];   \
                 i##_D5_AXIS++) {                                              \
              uint64_t srcAddr = i0 * eBatchPitch[0] + i1 * eBatchPitch[1] +   \
                                 i2 * eBatchPitch[2] + i3 * eBatchPitch[3] +   \
                                 i4 * eBatchPitch[4] + i5 * eBatchPitch[5];    \
              sum += tBatch[srcAddr] - offset[0];                              \
            }                                                                  \
            size_t i##_D5_AXIS = 0;                                            \
            int32_t res = nearbyintf(sum * scale[0] * invScale) + offset[1];   \
            uint64_t dstAddr = i0 * eDstPitch[0] + i1 * eDstPitch[1] +         \
                               i2 * eDstPitch[2] + i3 * eDstPitch[3] +         \
                               i4 * eDstPitch[4] + i5 * eDstPitch[5];          \
            tOutput[dstAddr] = clip<int32_t, int8_t>(res);                     \
          }
  // Each loop order, with the inner-most dimension/index equal to the axis.
  switch (axis) {
  case 0:
    LOOP_AXIS_CASE(1, 2, 3, 4, 5, 0);
    break;
  case 1:
    LOOP_AXIS_CASE(0, 2, 3, 4, 5, 1);
    break;
  case 2:
    LOOP_AXIS_CASE(0, 1, 3, 4, 5, 2);
    break;
  case 3:
    LOOP_AXIS_CASE(0, 1, 2, 4, 5, 3);
    break;
  case 4:
    LOOP_AXIS_CASE(0, 1, 2, 3, 5, 4);
    break;
  case 5:
    LOOP_AXIS_CASE(0, 1, 2, 3, 4, 5);
    break;
  default: // TODO Add some warning message(axis bigger than num of dims)
    break;
  }
#undef LOOP_AXIS_CASE
}


void dnn_lib::fwdLibBatchedReduceAddInstInt8Threaded(
    void *pdst, void *pdstDims, void *pdstPitches, void *pbatch,
    void *pbatchDims, void *pbatchPitches, unsigned int pbatchDimNum,
    unsigned int axis, float *scale, int32_t *offset, uint64_t flags) {


  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;  

  int8_t *tOutput = (int8_t *)pdst;
  int8_t *tBatch = (int8_t *)pbatch;

  float invScale;
  getReciprocal(scale[1], invScale);
  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *batchIndex = (unsigned int *)pbatchDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *batchPitch = (unsigned int *)pbatchPitches;

  unsigned int numElemsDst;
  
  numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  getUniformCachelinePartition(sizeof(int8_t), numElemsDst, initialAddr, maxRead,
                               activeMinions);

  if (maxRead == 0)
    return;

  unsigned int offsets[pbatchDimNum - 1];

  unsigned int k; 

  unsigned int redBatchPitch[pbatchDimNum - 1];
  for (size_t i = 0; i < pbatchDimNum; i++) {
    if (i < axis) {
      redBatchPitch[i] = batchPitch[i];
      
    } else if (i > axis) {
      redBatchPitch[i - 1] = batchPitch[i];
    }
  }

  getNonPaddingCoordinates(offsets, initialAddr, pbatchDimNum - 1, dstPitch, dstIndex,
                           k);
  uint64_t offsetOut = 0;
  uint64_t offsetIn = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * offsets[j];
    offsetIn += redBatchPitch[j] * offsets[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (not done && offsetOut < posMax) {
    float sum = 0.0;
    for (size_t i = 0; i < batchIndex[axis]; i++) {
      // print(__PRETTY_FUNCTION__);
      sum += tBatch[offsetIn] - offset[0];
      offsetIn += batchPitch[axis];
    }
    offsetIn -= batchIndex[axis] * batchPitch[axis];
    int32_t res = nearbyintf(sum * scale[0] * invScale) + offset[1];   
    tOutput[offsetOut] = clip<int32_t, int8_t>(res); 
 
    done = getOffsets(pbatchDimNum - 1, offsets, offsetIn, offsetOut, dstIndex,
                      redBatchPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * sizeof(int8_t) / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + sizeof(int8_t)*initialAddr, clperminion);
}

// This version does NOT support Tensors of more than 2 dimensions with padding
template <typename srcType>
void dnn_lib::fwdLibSparseLengthsWeightedSumInst(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(pdst, scale[4], offset[4]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  const Addresser<srcType> tWInput(pweights, scale[1], offset[1]);
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  size_t segments = pLengthsSize;
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += lengths[i];
  }

  size_t totalSize = 1;
  for (size_t i = 0; i < pdstDimNum; i++) {
    totalSize *= dataIndex[i];
  }
  size_t lineSize = totalSize / dataIndex[0];

  // Output tensor should be zero at the begin
  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    // NOTE : Not C++ compliant?  Fails with clang.
    // float tmp[lineSize] = { 0.0f };
    float tmp[lineSize];
    for (size_t j = 0; j < lineSize; j++)
      tmp[j] = 0.0f;
    for (size_t j = 0, e = lengths[i]; j < e; j++) {
      float weight = tWInput[curIdx * weightPitch[0]];
      size_t offsetIn = indices[curIdx] * dataPitch[0];
      for (size_t k = 0; k < lineSize; k++) {
        tmp[k] += tAInput[offsetIn] * weight;
        offsetIn++;
      }
      curIdx++;
    }
    size_t offsetOut = i * dstPitch[0];
    for (size_t k = 0; k < lineSize; k++) {
      tOutput[offsetOut] = tmp[k];
      offsetOut++;
    }
  }
}

// This version DOES support Tensors of more than 2 dimensions with padding
template <typename srcType>
void dnn_lib::fwdLibSparseLengthsWeightedSumInstThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize, float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(pdst, scale[4], offset[4]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  const Addresser<srcType> tWInput(pweights, scale[1], offset[1]);
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  size_t segments = pLengthsSize;
  size_t ranges[segments];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    ranges[i] = totalLength;
    totalLength += lengths[i];
  }

  unsigned int coord[pdstDimNum];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, pdstDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++)
    offsetOut += coord[i] * dstPitch[i];
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    size_t segment_begin = ranges[coord[0]];
    size_t segment_end = segment_begin + lengths[coord[0]];

    size_t offsetIn = 0;
    for (int i = 1; i < pdstDimNum; i++)
      offsetIn += coord[i] * dataPitch[i];

    float res = 0;
    for (size_t k = segment_begin; k < segment_end; k++) {
      res += tAInput[indices[k] * dataPitch[0] + offsetIn] *
             (float)tWInput[k * weightPitch[0]];
    }

    tOutput[offsetOut] = res;
    done = getOffsets(pdstDimNum, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + typeSize*initialAddr, clperminion);
}

void dnn_lib::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
    void *pweightsDims, void *pweightsPitches, void *pindices, void *plengths,
    unsigned int pLengthsSize) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  float *tOutput = (float *)pdst;
  uint8_t *tAInput = (uint8_t *)pdata;
  float *tWInput = (float *)pweights;
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  size_t segments = pLengthsSize;
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += lengths[i];
  }
  // assert(totalLength == weightIndex[0] && "sum(Lengths) must be equal to
  // len(Indices)");

  size_t totalSizeIn = 1;
  size_t totalSizeOut = 1;
  for (size_t i = 0; i < pdstDimNum; i++) {
    totalSizeIn *= dataIndex[i];
    totalSizeOut *= dstIndex[i];
  }
  const size_t inLineSize = totalSizeIn / dataIndex[0];
  const size_t outLineSize = totalSizeOut / dstIndex[0];

  // Output tensor should be zero at the begin
  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    for (size_t j = 0, e = lengths[i]; j < e; j++) {
      const float weight = tWInput[curIdx * weightPitch[0]];
      size_t offsetIn = indices[curIdx] * dataPitch[0];
      size_t offsetOut = i * dstPitch[0];
      curIdx++;
      // Get the scale and offset from the row; go to the current row and offset
      // into it up until the last 8 bytes. Use memcpy to get the values out to
      // avoid alignment issues of accessing 4-byte values.
      const unsigned char *currRowScaleOffsetPtr =
          &tAInput[0] + offsetIn + inLineSize * sizeof(uint8_t) - 8;
      float scale;
      float offset;
      memcpy(&scale, currRowScaleOffsetPtr, sizeof(float));
      memcpy(&offset, currRowScaleOffsetPtr + sizeof(float), sizeof(float));
      for (size_t k = 0; k < outLineSize; k++) {

        float d = dequantizeWithFloatOffset(tAInput[offsetIn], scale, offset);
        tOutput[offsetOut] += d * weight;
        offsetOut++;
        offsetIn++;
      }
    }
  }
}

template<typename DstType>
void dnn_lib::
    fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized(
        void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
        void *pdst2, void *pdst2Pitches, 
        void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
        void *pweightsDims, void *pweightsPitches, void *pindices,
        void *plengths, unsigned int pLengthsSize, uint64_t flags,
        const uint32_t minionOffset, const uint32_t assignedMinions) {

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (32 * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions)
    return;

  float *tOutput = (float *)pdst;
  uint16_t *tCvtOutput = (uint16_t *)pdst2;
  uint8_t *tAInput = (uint8_t *)pdata;
  float *tWInput = (float *)pweights;
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dst2Pitch = (unsigned int *)pdst2Pitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  size_t segments = pLengthsSize;
  size_t ranges[segments];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    ranges[i] = totalLength;
    totalLength += lengths[i];
  }

  size_t inLineSize = 1;
  size_t outLineSize = 1;
  for (size_t i = 1; i < pdstDimNum; i++) {
    inLineSize *= dataIndex[i];
    outLineSize *= dstIndex[i];
  }
  inLineSize -= 8;
  unsigned int numRegs = outLineSize/8;
  unsigned int lastMask = (1 << (((outLineSize-1) & 0x7) + 1)) - 1;

  unsigned int numElemsDst = dstPitch[0] * segments;
  unsigned int cll = 64 / sizeof(float);
  unsigned int rowsperminion = cll / dstPitch[0];
  unsigned int total_rows = rowsperminion * activeMinions;
  for (unsigned int i = total_rows; i < segments; i += activeMinions)
    rowsperminion++;
  unsigned int row_begin = minionId * rowsperminion;
  if (row_begin >= segments)
    return;
  unsigned int row_end = row_begin + rowsperminion;

  // Output tensor should be zero at the begin
  size_t curIdx = ranges[row_begin];
  for (size_t i = row_begin; i < row_end; i++) {
    for (size_t j = 0, e = lengths[i]; j < e; j++) {
      volatile uint8_t * data_ptr   = tAInput + indices[curIdx] * dataPitch[0];
      float            * scale_ptr  = (float *) &data_ptr[inLineSize];
      float            * offset_ptr = (float *) &data_ptr[inLineSize + 4];
      float            * weight_ptr = (float *) &tWInput[curIdx];
      float            * dst_ptr    = tOutput + i * dstPitch[0];
      uint16_t         * dst2_ptr   = tCvtOutput + i * dst2Pitch[0];

      volatile int32_t gather_offsets[] = { 0, 1, 2, 3, 4, 5, 6, 7 };

      if (std::is_same<DstType, float>::value)
      {
#undef LOAD_WEIGHT_SUM_AND_STORE
#define LOAD_WEIGHT_SUM_AND_STORE            \
            "fgb.ps f1, f31, %[data_ptr]\n"  \
            "flw.ps f2, 0x0(%[dst_ptr])\n"   \
            "fand.pi f1,  f1, f30\n"         \
            "fcvt.ps.pw f1,  f1\n"           \
            "fmadd.ps f1,  f1,  f29, f28\n"  \
            "fmadd.ps f0, f27, f1,  f2\n"    \
            "fsw.ps f0, 0x0(%[dst_ptr])\n"

        __asm__ __volatile__ (
              "mov.m.x m0, zero, 0xff\n"
              "fxor.pi f0, f0, f0\n"
              "li      t0, 0xff\n"
              "fbcx.ps f30, t0\n"
              "flw.ps  f31, 0x0(%[gather_offsets])\n"
              "fbc.ps  f27, 0x0(%[weight_ptr])\n"
              "fbc.ps  f28, 0x0(%[offset_ptr])\n"
              "fbc.ps  f29, 0x0(%[scale_ptr])\n"

              "add    t0, zero, zero\n"
              "ble    %[numRegs], t0, 2f\n"
              "1:\n"

              LOAD_WEIGHT_SUM_AND_STORE

              "addi   %[data_ptr], %[data_ptr], 8\n"
              "addi   %[dst_ptr], %[dst_ptr], 0x20\n"
              "addi    t0, t0, 0x1\n"
              "ble     t0, %[numRegs], 1b\n"
              "2:\n"

              "mov.m.x m0, %[lastMask], 0x0\n"

              LOAD_WEIGHT_SUM_AND_STORE

            : [data_ptr]   "+&r"   (data_ptr),
              [dst_ptr]    "+&r"   (dst_ptr) 
            : [gather_offsets] "r" (gather_offsets),
              [offset_ptr] "r"     (offset_ptr),
              [scale_ptr]  "r"     (scale_ptr),
              [weight_ptr] "r"     (weight_ptr),
              [numRegs]    "r"     (numRegs),
              [lastMask]   "r"     (lastMask)
            : "f0", "t0", "f27", "f28", "f29", "f30", "f31" 
          );
      }
   
      if (std::is_same<DstType, float16>::value)
      {
#undef LOAD_WEIGHT_SUM_AND_STORE
#define LOAD_WEIGHT_SUM_AND_STORE                \
            "fgb.ps      f1, f31, %[data_ptr]\n" \
            "flw.ps      f2, 0x0(%[dst_ptr])\n"  \
            "fand.pi     f1, f1,  f30\n"         \
            "fcvt.ps.pw  f1, f1\n"               \
            "fmadd.ps    f1, f1,  f29, f28\n"    \
            "fmadd.ps    f0, f27, f1,  f2\n"     \
            "fsw.ps      f0, 0x0(%[dst_ptr])\n"  \
            "fcvt.f16.ps f0, f0\n"               \
            "fsch.ps     f0, f26(%[dst2_ptr])\n"

        __asm__ __volatile__ (
              "mov.m.x m0, zero, 0xff\n"
              "fxor.pi f0, f0, f0\n"
              "li      t0, 0xff\n"
              "fbcx.ps f30, t0\n"
              "flw.ps  f31, 0x0(%[gather_offsets])\n"
              "fadd.pi f26, f31, f31\n"
              "fbc.ps  f27, 0x0(%[weight_ptr])\n"
              "fbc.ps  f28, 0x0(%[offset_ptr])\n"
              "fbc.ps  f29, 0x0(%[scale_ptr])\n"

              "add    t0, zero, zero\n"
              "ble    %[numRegs], t0, 2f\n"
              "1:\n"

              LOAD_WEIGHT_SUM_AND_STORE

              "addi   %[data_ptr], %[data_ptr], 8\n"
              "addi   %[dst_ptr],  %[dst_ptr],  0x20\n"
              "addi   %[dst2_ptr], %[dst2_ptr], 0x10\n"
              "addi    t0, t0, 0x1\n"
              "ble     t0, %[numRegs], 1b\n"
              "2:\n"

              "mov.m.x m0, %[lastMask], 0x0\n"

              LOAD_WEIGHT_SUM_AND_STORE

            : [data_ptr]   "+&r"   (data_ptr),
              [dst_ptr]    "+&r"   (dst_ptr),
              [dst2_ptr]   "+&r"   (dst2_ptr) 
            : [gather_offsets] "r" (gather_offsets),
              [offset_ptr] "r"     (offset_ptr),
              [scale_ptr]  "r"     (scale_ptr),
              [weight_ptr] "r"     (weight_ptr),
              [numRegs]    "r"     (numRegs),
              [lastMask]   "r"     (lastMask)
            : "f0", "t0", "f26", "f27", "f28", "f29", "f30", "f31" 
          );
      }

      curIdx++;
    }
  }
}

template<typename DstType>
void dnn_lib::
    fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyOptimized(
        void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
        void *pdata, void *pdataDims, void *pdataPitches, void *pweights,
        void *pweightsDims, void *pweightsPitches, void *pindices,
        void *plengths, unsigned int pLengthsSize, uint64_t flags,
        const uint32_t minionOffset, const uint32_t assignedMinions) {

  // Get offset of the Minion inside the group of Minions assigned to this Node.
  int64_t minionId = get_minion_id() - minionOffset;

  // Get number of Minions assigned to this Node.
  int64_t activeMinions = (assignedMinions == 0) ? (32 * ACTIVE_SHIRES) : assignedMinions;

  // If Minion is outside the group assigned to this Node get out.
  if ((minionId < 0) || (minionId >= activeMinions))
    return;

  // Set real types for input pointers.
  // For dst we used uint8_t because it can be accessed with different types.
  uint8_t *tOutput = (uint8_t *) pdst;
  uint8_t *tAInput = (uint8_t *) pdata;
  float   *tWInput = (float   *) pweights;
  int64_t *indices = (int64_t *) pindices;
  int32_t *lengths = (int32_t *) plengths;

  uint32_t *dstDims     = (uint32_t *) pdstDims;
  uint32_t *dataDims    = (uint32_t *) pdataDims;
  uint32_t *dstPitches  = (uint32_t *) pdstPitches;
  uint32_t *dataPitches = (uint32_t *) pdataPitches;

  // TODO : Add assert checking segments is equal to the number of output rows.

  // TODO : Add assert checking that totalLength is smaller than the size of
  // the indices tensor.
  //
  // Compute the total number of rows in data to be summed.
  //uintptr_t segments = pLengthsSize;
  //uintptr_t totalLength = 0;
  //for (uintptr_t i = 0; i < segments; i++)
  //  totalLength += lengths[i];
  //

  // Compute the number of elements per data row (first tensor dimension).
  uintptr_t dataRowSize = 1;
  for (uintptr_t i = 1; i < pdstDimNum; i++) dataRowSize *= dataDims[i];

  // Compute the number of elements per output row (first tensor dimension).
  uintptr_t dstRowSize = 1;
  for (uintptr_t i = 1; i < pdstDimNum; i++) dstRowSize *= dstDims[i];

  // Get size of the output element.
  uintptr_t dstElemSize = (std::is_same<DstType, float16>::value) ? 2 : 4;

  // Compute the number of 8-element vectors per output cache line.
  uintptr_t dstCacheLineVRegs = 64 / (dstElemSize * 8);

  // Compute the number of Cache Line groups per output row (rounded up).
  uintptr_t dstRowGroups = ((dstRowSize - 1) / 64) + 1;

  // Determine if row has a tail.
  bool dstRowHasTail = ((dstRowSize % 64) != 0);

  // Compute the number of 8-element vectors in the tail of the row.
  uintptr_t dstRowTailVRegs = (((dstRowSize - 1) / 8) + 1) % dstCacheLineVRegs;

  // Compute the element mask for the tail of the row.
  uint8_t dstRowTailVRegMask = (1 << (((dstRowSize - 1) % 8) + 1)) - 1;

  // Assign work to Minions :
  //
  // - Each Minion gets assigned at least one group of output cache lines
  //
  
  uintptr_t totalWorkUnits = dstRowGroups * dstDims[0];

  //  Distribute the tail of groups.
  uintptr_t minionWorkUnits;
  if ((totalWorkUnits % activeMinions) == 0) {
    minionWorkUnits = totalWorkUnits / activeMinions;
  }
  else {
    minionWorkUnits = totalWorkUnits / activeMinions;
    uintptr_t remainingWorkUnits = totalWorkUnits % activeMinions;
    if (minionId < remainingWorkUnits)
      minionWorkUnits++;
  }

  // Compute the index into the first work unit.
  uintptr_t minionFirstWorkUnit = minionId * minionWorkUnits;

  // Compute the first output row (segment) assigned to the Minion.
  uintptr_t minionFirstSegment = minionFirstWorkUnit / dstRowGroups;

  // Compute current group in row assigned to the Minion.
  uintptr_t minionFirstRowGroup = minionFirstWorkUnit % dstRowGroups;

  // Get the first index assigned to the Minion.
  uintptr_t minionFirstIndex = 0;
  for (uintptr_t i = 0; i < minionFirstSegment; i++)
    minionFirstIndex += lengths[i];

  // Initialize indices. 
  uintptr_t minionCurrIndex = minionFirstIndex;
  uintptr_t minionCurrSegment = minionFirstSegment;
  uintptr_t minionCurrRowGroup = minionFirstRowGroup;
  uintptr_t currSegmentLength = lengths[minionCurrSegment];
  
  // Initilize output pointer.
  uint8_t *dst_ptr = tOutput + (minionCurrSegment * dstPitches[0] + minionCurrRowGroup * 64) * dstElemSize; 

  // For all minion assigned work units
  for (uintptr_t i = 0; i < minionWorkUnits; i++) {

    // Detect row tail
    bool dstGroupNotInRowTail = !dstRowHasTail || (minionCurrRowGroup != (dstRowGroups - 1));

    if (dstGroupNotInRowTail) {
      // Not in tail

      volatile int32_t gather_offsets[] = { 0, 1, 2, 3, 4, 5, 6, 7 };

      // Initialize vector mask
      // Clear vector registers that will be used for accumulation
      // Initialize offsets for gather from input
      // Initialize mask to clear upper bytes from input load
      __asm__ __volatile__ (
        "mov.m.x m0, zero, 0xff\n"
        "fxor.pi f0, f0, f0\n"
        "fxor.pi f1, f0, f0\n"
        "fxor.pi f2, f0, f0\n"
        "fxor.pi f3, f0, f0\n"
        "fxor.pi f4, f0, f0\n"
        "fxor.pi f5, f0, f0\n"
        "fxor.pi f6, f0, f0\n"
        "fxor.pi f7, f0, f0\n"
        "li      t0, 0xff\n"
        "fbcx.ps f30, t0\n"
        "flw.ps  f31, 0x0(%[gather_offsets])\n"
        : 
        : [gather_offsets] "r" (gather_offsets)
        : "t0", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f30", "f31"
      );

      // For all sparse input rows.
      for (uintptr_t j = 0, currIndex = minionCurrIndex;
           j < currSegmentLength; j++, currIndex++) {
        volatile uint8_t * data_ptr   = tAInput + indices[currIndex] * dataPitches[0];
        float            * scale_ptr  = (float *) &data_ptr[dataRowSize - 8];
        float            * offset_ptr = (float *) &data_ptr[dataRowSize - 4];
        float            * weight_ptr = (float *) &tWInput[currIndex];

        __asm__ __volatile__ (
          "fbc.ps  f27, 0x0(%[weight_ptr])\n"
          "fbc.ps  f28, 0x0(%[offset_ptr])\n"
          "fbc.ps  f29, 0x0(%[scale_ptr])\n"

          // Load a full input cache line (64 elements, 8 vregs)
          "fgb.ps     f26, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
          "fgb.ps     f25, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
          "fgb.ps     f24, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
          "fgb.ps     f23, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
          "fgb.ps     f22, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
          "fgb.ps     f21, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
          "fgb.ps     f20, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
          "fgb.ps     f19, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
          "fand.pi    f26, f26, f30\n"
          "fand.pi    f25, f25, f30\n"
          "fand.pi    f24, f24, f30\n"
          "fand.pi    f23, f23, f30\n"
          "fand.pi    f22, f22, f30\n"
          "fand.pi    f21, f21, f30\n"
          "fand.pi    f20, f20, f30\n"
          "fand.pi    f19, f19, f30\n"
          "fcvt.ps.pw f26, f26\n"
          "fcvt.ps.pw f25, f25\n"
          "fcvt.ps.pw f24, f24\n"
          "fcvt.ps.pw f23, f23\n"
          "fcvt.ps.pw f22, f22\n"
          "fcvt.ps.pw f21, f21\n"
          "fcvt.ps.pw f20, f20\n"
          "fcvt.ps.pw f19, f19\n"
          "fmadd.ps   f26, f26, f29, f28\n"
          "fmadd.ps   f25, f25, f29, f28\n"
          "fmadd.ps   f24, f24, f29, f28\n"
          "fmadd.ps   f23, f23, f29, f28\n"
          "fmadd.ps   f22, f22, f29, f28\n"
          "fmadd.ps   f21, f21, f29, f28\n"
          "fmadd.ps   f20, f20, f29, f28\n"
          "fmadd.ps   f19, f19, f29, f28\n"
          "fmadd.ps   f0, f27, f26, f0\n"
          "fmadd.ps   f1, f27, f25, f1\n"
          "fmadd.ps   f2, f27, f24, f2\n"
          "fmadd.ps   f3, f27, f23, f3\n"
          "fmadd.ps   f4, f27, f22, f4\n"
          "fmadd.ps   f5, f27, f21, f5\n"
          "fmadd.ps   f6, f27, f20, f6\n"
          "fmadd.ps   f7, f27, f19, f7\n"
         : [data_ptr]   "+&r" (data_ptr)
         : [offset_ptr] "r"   (offset_ptr),
           [scale_ptr]  "r"   (scale_ptr),
           [weight_ptr] "r"   (weight_ptr)
         : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
           "f19", "f20", "f21", "f22", "f23", "f24", "f25",
           "f26", "f27", "f28", "f29"
        );
      }

      // Store accumulated results.
      __asm__ __volatile__ (
        "fsw.ps f0,    (%[dst_ptr])\n"
        "fsw.ps f1,  32(%[dst_ptr])\n"
        "fsw.ps f2,  64(%[dst_ptr])\n"
        "fsw.ps f3,  96(%[dst_ptr])\n"
        "fsw.ps f4, 128(%[dst_ptr])\n"
        "fsw.ps f5, 160(%[dst_ptr])\n"
        "fsw.ps f6, 192(%[dst_ptr])\n"
        "fsw.ps f7, 224(%[dst_ptr])\n"
        :
        : [dst_ptr] "r" (dst_ptr)
        :
      );

      minionCurrRowGroup++;

      minionCurrIndex += currSegmentLength;

      dst_ptr += 64 * dstElemSize;
    }
    else {
      volatile int32_t gather_offsets[] = { 0, 1, 2, 3, 4, 5, 6, 7 };

      // Initialize vector mask
      // Clear vector registers that will be used for accumulation
      // Initialize offsets for gather from input
      // Initialize mask to clear upper bytes from input load
      __asm__ __volatile__ (
        "mov.m.x m0, zero, 0xff\n"
        "fxor.pi f0, f0, f0\n"
        "li      t0, 0xff\n"
        "fbcx.ps f30, t0\n"
        "flw.ps  f31, 0x0(%[gather_offsets])\n"
        : 
        : [gather_offsets] "r" (gather_offsets)
        : "t0", "f0", "f30", "f31"
      );

      for (uintptr_t k = 0; k < (dstRowTailVRegs - 1); k++) {
    
        // For all sparse input rows.
        for (uintptr_t j = 0, currIndex = minionCurrIndex;
             j < currSegmentLength; j++, currIndex++) {
          volatile uint8_t * data_ptr   = tAInput + indices[currIndex] * dataPitches[0];
          float            * scale_ptr  = (float *) &data_ptr[dataRowSize - 8];
          float            * offset_ptr = (float *) &data_ptr[dataRowSize - 4];
          float            * weight_ptr = (float *) &tWInput[currIndex];

          __asm__ __volatile__ (
            "fbc.ps  f27, 0x0(%[weight_ptr])\n"
            "fbc.ps  f28, 0x0(%[offset_ptr])\n"
            "fbc.ps  f29, 0x0(%[scale_ptr])\n"

            // Load a full input cache line (64 elements, 8 vregs)
            "fgb.ps     f26, f31, %[data_ptr]\n"
            "addi       %[data_ptr], %[data_ptr], 8\n"
            "fand.pi    f26, f26, f30\n"
            "fcvt.ps.pw f26, f26\n"
            "fmadd.ps   f26, f26, f29, f28\n"
            "fmadd.ps   f0, f27, f26, f0\n"
           : [data_ptr]   "+&r" (data_ptr)
           : [offset_ptr] "r"   (offset_ptr),
             [scale_ptr]  "r"   (scale_ptr),
             [weight_ptr] "r"   (weight_ptr)
           : "f0", "f26", "f27", "f28", "f29"
          );
        }

        // Store accumulated results.
        __asm__ __volatile__ (
          "fsw.ps f0, (%[dst_ptr])\n"
          :
          : [dst_ptr] "r" (dst_ptr)
          :
        );
      }

      // Set mask for last VReg in group.
      __asm__ __volatile__ (
        "mov.m.x m0, %[tail_mask], 0x0\n"
        :
        : [tail_mask] "r" (dstRowTailVRegMask)
      );

      // For all sparse input rows.
      for (uintptr_t j = 0, currIndex = minionCurrIndex;
           j < currSegmentLength; j++, currIndex++) {
        volatile uint8_t * data_ptr   = tAInput + indices[currIndex] * dataPitches[0];
        float            * scale_ptr  = (float *) &data_ptr[dataRowSize - 8];
        float            * offset_ptr = (float *) &data_ptr[dataRowSize - 4];
        float            * weight_ptr = (float *) &tWInput[currIndex];

        __asm__ __volatile__ (
          "fbc.ps  f27, 0x0(%[weight_ptr])\n"
          "fbc.ps  f28, 0x0(%[offset_ptr])\n"
          "fbc.ps  f29, 0x0(%[scale_ptr])\n"

          // Load a full input cache line (64 elements, 8 vregs)
          "fgb.ps     f26, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
          "fand.pi    f26, f26, f30\n"
          "fcvt.ps.pw f26, f26\n"
          "fmadd.ps   f26, f26, f29, f28\n"
          "fmadd.ps   f0, f27, f26, f0\n"
         : [data_ptr]   "+&r" (data_ptr)
         : [offset_ptr] "r"   (offset_ptr),
           [scale_ptr]  "r"   (scale_ptr),
           [weight_ptr] "r"   (weight_ptr)
         : "f0", "f26", "f27", "f28", "f29"
        );
      }

      // Store accumulated results.
      __asm__ __volatile__ (
        "fsw.ps f0, (%[dst_ptr])\n"
        "mov.m.x m0, zero, 0xff\n"
        :
        : [dst_ptr] "r" (dst_ptr)
        :
      );

      minionCurrIndex += currSegmentLength;

      // Move from row tail to next row.
      minionCurrSegment++;
      minionCurrRowGroup = 0;
      currSegmentLength = lengths[minionCurrSegment];

      dst_ptr = tOutput + (minionCurrSegment * dstPitches[0] + minionCurrRowGroup * 64) * dstElemSize;
    }

  }
}

void dnn_lib::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  float *tOutput = (float *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  float *tScale = (float *)pscale;
  float *tOffset = (float *)poffset;
  float *tWInput = (float *)pweights;
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  size_t segments = pLengthsSize;
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    totalLength += lengths[i];
  }
  // assert(totalLength == weightIndex[0] && "sum(Lengths) must be equal to
  // len(Indices)");

  size_t totalSize = 1;
  for (size_t i = 0; i < pdstDimNum; i++) {
    totalSize *= dataIndex[i];
  }
  size_t lineSize = totalSize / dataIndex[0];

  // Output tensor should be zero at the begin
  size_t curIdx = 0;
  for (size_t i = 0; i < segments; i++) {
    for (size_t j = 0, e = lengths[i]; j < e; j++) {
      const float weight = tWInput[curIdx * weightPitch[0]];
      const size_t rowIdx = indices[curIdx];
      const float scale = tScale[rowIdx];
      const float offset = tOffset[rowIdx];
      size_t offsetIn = rowIdx * dataPitch[0];
      size_t offsetOut = i * dstPitch[0];
      curIdx++;
      for (size_t k = 0; k < lineSize; k++) {

        float d = dequantizeWithFloatOffset(tAInput[offsetIn], scale, offset);
        tOutput[offsetOut] += d * weight;
        offsetOut++;
        offsetIn++;
      }
    }
  }
}

/*
void dnn_lib::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags) {
  float *tOutput = (float *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  float *tScale = (float *)pscale;
  float *tOffset = (float *)poffset;
  float *tWInput = (float *)pweights;
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];
  unsigned int initialAddr, maxRead;
  getCachelinePartition(sizeof(float), numElemsDst, initialAddr, maxRead,
activeMinions);

  size_t segments = pLengthsSize;
  size_t ranges[segments];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    ranges[i] = totalLength;
    totalLength += lengths[i];
  }

  unsigned int coord[pdstDimNum];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, pdstDimNum, dstPitch, dstIndex,
k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) offsetOut += coord[i]*dstPitch[i];
  if (offsetOut >= numElemsDst) return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  while(!done) {
    size_t segment_begin = ranges[coord[0]];
    size_t segment_end = segment_begin + lengths[coord[0]];

    size_t offsetIn = 0;
    for (int i = 1; i < pdstDimNum; i++) offsetIn += coord[i] * dataPitch[i];

    float res = 0;
    for (size_t k = segment_begin; k < segment_end; k++) {
      size_t idx = indices[k];
      float d = dequantizeWithFloatOffset(tAInput[indices[k]*dataPitch[0] +
offsetIn], tScale[k], tOffset[k]); res += d * (float)tWInput[k *
weightPitch[0]];
    }

    tOutput[offsetOut] = res;

    done = getOffsets(pdstDimNum, coord, offsetOut, dstIndex, dstPitch);
    if (offsetOut >= posMax) break;
  }
}
*/

void dnn_lib::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  float *tOutput = (float *)pdst;
  int8_t *tAInput = (int8_t *)pdata;
  float *tScale = (float *)pscale;
  float *tOffset = (float *)poffset;
  float *tWInput = (float *)pweights;
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *weightIndex = (unsigned int *)pweightsDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;
  unsigned int *weightPitch = (unsigned int *)pweightsPitches;

  size_t segments = pLengthsSize;
  size_t ranges[segments];
  size_t totalLength = 0;
  for (size_t i = 0; i < segments; i++) {
    ranges[i] = totalLength;
    totalLength += lengths[i];
  }
  // assert(totalLength == weightIndex[0] && "sum(Lengths) must be equal to
  // len(Indices)");

  size_t lineSize = 1;
  for (size_t i = 1; i < pdstDimNum; i++)
    lineSize *= dataIndex[i];

  unsigned int numElemsDst = dstPitch[0] * segments;
  unsigned int cll = 64 / sizeof(float);
  unsigned int rowsperminion = (cll - 1) / dstPitch[0] + 1;
  unsigned int total_rows = rowsperminion * activeMinions;
  for (unsigned int i = total_rows; i < segments; i += activeMinions)
    rowsperminion++;
  unsigned int row_begin = minionId * rowsperminion;
  if (row_begin >= segments)
    return;
  unsigned int row_end = row_begin + rowsperminion;

  size_t curIdx = ranges[row_begin];
  for (size_t i = row_begin; i < row_end; i++) {
    for (size_t j = 0, e = lengths[i]; j < e; j++) {
      const float weight = tWInput[curIdx * weightPitch[0]];
      const size_t rowIdx = indices[curIdx];
      const float scale = tScale[rowIdx];
      const float offset = tOffset[rowIdx];
      size_t offsetIn = rowIdx * dataPitch[0];
      size_t offsetOut = i * dstPitch[0];
      curIdx++;
      for (size_t k = 0; k < lineSize; k++) {

        float d = dequantizeWithFloatOffset(tAInput[offsetIn], scale, offset);
        tOutput[offsetOut] += d * weight;
        offsetOut++;
        offsetIn++;
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibLengthsSumInst(void *pdst, void *pdstDims,
                                   void *pdstPitches, void *pdata,
                                   void *pdataDims, void *pdataPitches,
                                   unsigned int pdataDimNum, void *plengths,
                                   unsigned int pLengthsSize, float *scale,
                                   int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId > 0)
    return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tTmp(pdst, scale[2], offset[2]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eDataPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < pdataDimNum; i++) {
    eDims[i] = dataIndex[i];
    eDstPitch[i] = dstPitch[i];
    eDataPitch[i] = dataPitch[i];
  }

  uint64_t addrSrc, addrDst;
  // Global index inside data
  size_t posIn = 0;
  for (size_t i = 0; i < pLengthsSize; i++) {
    for (int32_t j = 0, e = lengths[i]; j < e; j++, posIn++) {
      // Sum elements across batch dimension
      for (size_t y = 0; y < eDims[1]; y++) {
        for (size_t z = 0; z < eDims[2]; z++) {
          for (size_t w = 0; w < eDims[3]; w++) {
            for (size_t q = 0; q < eDims[4]; q++) {
              for (size_t r = 0; r < eDims[5]; r++) {
                addrDst = i * eDstPitch[0] + y * eDstPitch[1] +
                          z * eDstPitch[2] + w * eDstPitch[3] +
                          q * eDstPitch[4] + r * eDstPitch[5];
                addrSrc = posIn * eDataPitch[0] + y * eDataPitch[1] +
                          z * eDataPitch[2] + w * eDataPitch[3] +
                          q * eDataPitch[4] + r * eDataPitch[5];
                tOutput[addrDst] = tTmp[addrDst] + tAInput[addrSrc];
              }
            }
          }
        }
      }
    }
  }
}



template <typename srcType>
void dnn_lib::fwdLibLengthsSumInstThreaded(void *pdst, void *pdstDims,
                                           void *pdstPitches, void *pdata,
                                           void *pdataDims, void *pdataPitches,
                                           unsigned int pdataDimNum, void *plengths,
                                           unsigned int pLengthsSize, float *scale,
                                           int32_t *offset, uint64_t flags) {
  
  unsigned int minion_id = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minion_id >= activeMinions) return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tTmp(pdst, scale[2], offset[2]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  int32_t *lengths = (int32_t *)plengths;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0]; // Total number of elements in the tensor
  
  // We give to each minion an initial address and the number of positions that it must work on (maxRead).
  unsigned int initialAddr, maxRead;
  getCachelinePartition(sizeof(srcType), numElemsDst, initialAddr, maxRead, minion_id, activeMinions);
  if (maxRead == 0) return;

  // We move the initialAddr to the next non-padding position

  unsigned int k; // Amount of non-zero coordinates
  unsigned int coordIn[pdataDimNum];  // Vector of coordinates
  unsigned int coordOut[pdataDimNum]; // Vector of coordinates

  getNonPaddingCoordinates(coordOut, initialAddr, pdataDimNum, dstPitch, dstIndex, k);
  for (unsigned int i = 1; i < pdataDimNum; i++) {
    coordIn[i] = coordOut [i];
  }
  for (unsigned int l = 0; l < coordOut[0]; l++) coordIn[0] += lengths[l]; 

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += dataPitch[j]*coordIn[j];
    offsetOut += dstPitch[j]*coordOut[j];
  }
  unsigned int offsetIn0 = offsetIn;
  unsigned int offsetOut0 = offsetOut;
  unsigned int offsetOutmax;

  unsigned int coordIn0[pdataDimNum]; 
  unsigned int coordOut0[pdataDimNum];

  for (unsigned int i = 0; i < pdataDimNum; i++) {
    coordIn0[i] = coordIn[i];
    coordOut0[i] = coordOut[i];
  }

  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until completion.
  bool endmatrix = false;
  bool done = false;
  size_t posIn = 0;
  while (!done) {
    for (size_t posIn = 0; posIn < lengths[coordOut[0]]; posIn++) {
      while (!endmatrix && (offsetOut < posMax)) {
        tOutput[offsetOut] = tAInput[offsetIn] + tTmp[offsetOut];
        for (size_t j = pdataDimNum - 1; j > 0; j--) {
          if (coordIn[j] != (dataIndex[j] - 1)) {
            offsetIn += dataPitch[j];
            offsetOut += dstPitch[j];
            coordIn[j]++;
            coordOut[j]++;
            break;
          }
          else if (j != 1) {
            offsetIn -= (dataIndex[j] - 1) * dataPitch[j];
            offsetOut -= (dstIndex[j] - 1) * dstPitch[j];
            coordIn[j] = 0;
            coordOut[j] = 0;
          }
          else {
            if (coordOut[0] == dstIndex[0] - 1) {
              done = true;
             // tOutput[offsetOut] = -2;
            }
            endmatrix = true;
            break;
          }
        }
      }
      if (offsetOut >= posMax) {
        done = true;
      //  tOutput[offsetOut] = offsetOut0;
      //  tOutput[offsetOut + 1] = offsetOut;
      }
      offsetIn = offsetIn0 + (posIn + 1) * dataPitch[0];
      offsetOut = offsetOut0;

      for (unsigned int i = 0; i < pdataDimNum; i++) {
        coordIn[i] = coordIn0[i];
        coordOut[i] = coordOut0[i];
      }     
      coordIn[0] += (posIn + 1);
      endmatrix = false;
    }
    if (done) {
     // tOutput[offsetOut] = -10;
      break;
    }
    offsetIn = 0;
    offsetIn0 = 0;
    offsetOut = 0;
    offsetOut0 = 0;
    for (unsigned int i = 1; i < pdataDimNum; i++) {
      coordIn[i] = coordIn0[i] = 0;
      coordOut[i] = coordOut0[i] = 0;
    }     
    for (int j = 0; j <= coordOut0[0]; j++) {
      offsetIn += dataPitch[0] * lengths[j];
      offsetIn0 += dataPitch[0] * lengths[j];
      offsetOut += dstPitch[0];
      offsetOut0 += dstPitch[0];
    }   
    coordIn[0] += lengths[coordOut0[0]];
    coordIn0[0] += lengths[coordOut0[0]];
    coordOut[0]++;
    coordOut0[0]++;
  }

  if (!DO_EVICTS) return;
  unsigned int clperminion = maxRead*sizeof(srcType)/64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)pdst + sizeof(srcType)*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibSparseToDenseInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *srcT,
                                      void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, void *indicesT,
                                      void *indDims, void *indPitches,
                                      float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId > 0)
    return;

  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);
  const Addresser<srcType> tTmp(dstT, scale[2], offset[2]);
  long long *tIndex = (long long *)indicesT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;
  unsigned int *indIndex = (unsigned int *)indDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;
  unsigned int *indPitch = (unsigned int *)indPitches;

  // Convert sparse representation to dense representation by taking
  // slices of output and values and accumulating the value slice into
  // the output slice.

  // Dimensions and coord for the output and values slices. sliceDims
  // will always be {1, [rest of output dimensions]} since the first dimension
  // is the index in this operation. sliceOffsets will be {indices[j], 0, ...}
  // for the output slice and {j, 0, ...} for the values slice so that the
  // slice at index j gets mapped to index indices[j] in the dense
  // representation.

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = dstIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }
  eBatchDims[0] = 1;
  uint64_t addrSrc, addrDst;
  for (size_t j = 0; j < indIndex[0]; j++) {
    long long index = tIndex[j];
    // We can use this loop for all shapes.
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc = j * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = index * eDstPitch[0] + y * eDstPitch[1] +
                        z * eDstPitch[2] + w * eDstPitch[3] + q * eDstPitch[4] +
                        r * eDstPitch[5];
              tOutput[addrDst] = tTmp[addrDst] + tInput[addrSrc];
            }
          }
        }
      }
    }
  }
}



template <typename srcType>
void dnn_lib::fwdLibSparseToDenseInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, void *indicesT, void *indDims,
    void *indPitches, float *scale, int32_t *offset, uint64_t flags) {
  
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);
  long long *tIndex = (long long *)indicesT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;
  unsigned int *indIndex = (unsigned int *)indDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;
  unsigned int *indPitch = (unsigned int *)indPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    coord[i] = 0;
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetIn = 0; // Doesn't include srcPitch[0]. offsetIn doesn't
                             // have the conventional meaning
  unsigned int offsetOut = 0;

  unsigned int srcPitch_0 = srcPitch[0];
  srcPitch[0] = 0;

  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
    offsetIn += srcPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
 
  while (!done && (offsetOut < posMax)) {
    srcType sum;
    sum = 0;
    for (size_t j = 0; j < indIndex[0]; j++) {
      if (tIndex[j] == coord[0]) sum = sum + tInput[offsetIn + j * srcPitch_0];
    }
    tOutput[offsetOut] = sum;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, dstIndex,
                      srcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value, 
std::size_t>::type = 0>
void sparseToDenseOp (uintptr_t dst, uintptr_t src, long long* tIndex, unsigned int batchPitch, 
unsigned int batch, unsigned int numIndices, size_t typeSize, float *scale, int32_t *offset){
  
  volatile int32_t gatherValues[] = {0, 4, 8, 12, 16, 20, 24, 28};
  __asm__ __volatile__("add t0, zero, zero\n"
                       "fxor.pi f0, f0, f0\n"

                       "addi    t3, %[tIndex], 0x0\n"
                       "flw.ps f31, 0x0(%[gatherValues])\n"
                       "1:\n"
                            
                       "ld t1, 0x0(t3)\n"
                       "bne t1, %[batch], 2f\n"

                       "mul t2, t0, %[typeSize]\n"
                       "mul t2, t2, %[batchPitch]\n"
                       "add t2, t2, %[src]\n"
                       "fgw.ps  f1, f31(t2) \n"
                       "fadd.ps f0, f0, f1 \n"
                       "2:\n"

                       "addi t0, t0, 0x1\n"
                       "addi t3, t3, 0x8\n"
                       "blt t0, %[numIndices], 1b\n"

                       "fscw.ps  f0, f31(%[dst]) \n"

                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ src ] "r"(src),
                         [ numIndices ] "r"(numIndices),
                         [ batch ] "r"(batch),
                         [ tIndex ] "r"(tIndex),
                         [ batchPitch ] "r"(batchPitch),
                         [ typeSize ] "r"(typeSize),
                         [ dst ] "r"(dst)                               
                       : "t0", "t1", "t2", "t3", "f0", "f1", "f31", "memory"); 

}

template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, 
std::size_t>::type = 0>
void sparseToDenseOp (uintptr_t dst, uintptr_t src, long long* tIndex, unsigned int batchPitch, 
unsigned int batch, unsigned int numIndices, size_t typeSize, float *scale, int32_t *offset){
  volatile int32_t gatherValues[] = {0, 2, 4, 6, 8, 10, 12, 14};

  
  __asm__ __volatile__("add t0, zero, zero\n"
                       "fxor.pi f0, f0, f0\n"
                       "fcvt.ps.f16 f0, f0\n"
                       "addi    t3, %[tIndex], 0x0\n"
                       "flw.ps f31, 0x0(%[gatherValues])\n" 
                       "1:\n"
                            
                       "ld t1, 0x0(t3)\n"
                       "bne t1, %[batch], 2f\n"

                       "mul t2, t0, %[typeSize]\n"
                       "mul t2, t2, %[batchPitch]\n"
                       "add t2, t2, %[src]\n"
                       "fgh.ps  f1, f31(t2)\n"
                       "fcvt.ps.f16 f1, f1\n"
                       "fadd.ps f0, f0, f1 \n"
                       "2:\n"

                       "addi t0, t0, 0x1\n"
                       "addi t3, t3, 0x8\n"
                       "ble t0, %[numIndices], 1b\n"

                       "fcvt.f16.ps f0, f0\n"
                       "fsch.ps  f0, f31(%[dst]) \n"

                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ src ] "r"(src),
                         [ numIndices ] "r"(numIndices),
                         [ batch ] "r"(batch),
                         [ tIndex ] "r"(tIndex),
                         [ batchPitch ] "r"(batchPitch),
                         [ typeSize ] "r"(typeSize),
                         [ dst ] "r"(dst)                               
                       : "t0", "t1", "t2", "t3", "f0", "f1", "f31", "memory"); 

}



template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value,
std::size_t>::type = 0>
void sparseToDenseOp (uintptr_t dst, uintptr_t src, long long* tIndex, unsigned int batchPitch, 
unsigned int batch, unsigned int numIndices, size_t typeSize, float *scale, int32_t *offset){
  volatile int32_t gatherValues[] = {0, 1, 2, 3, 4, 5, 6, 7};
  __asm__ __volatile__("add t0, zero, zero\n"
                       "fxor.pi f0, f0, f0\n"
                       "flw.ps f31, 0x0(%[gatherValues])\n"
                       "fbc.ps f30, 0x0(%[offset]) \n"
                       "fbc.ps f29, 0x0(%[scale]) \n"
           
                       "fsub.pi f0, f0, f30 \n"
                       "fcvt.ps.pw f0, f0 \n"
                       "fmul.ps f0, f0, f29 \n"  
      

                       "addi    t3, %[tIndex], 0x0\n"
                       "1:\n"
                            
                       "ld t1, 0x0(t3)\n"
                       "bne t1, %[batch], 2f\n"

                       "mul t2, t0, %[typeSize]\n"
                       "mul t2, t2, %[batchPitch]\n"
                       "add t2, t2, %[src]\n"
                      
                       "fgb.ps  f1, f31(t2) \n"
                       "fsub.pi f1, f1, f30 \n"
                       "fcvt.ps.pw f1, f1 \n"
                       
                       "fmadd.ps f0, f1, f29, f0 \n"

                       "2:\n"

                       "addi t0, t0, 0x1\n"
                       "addi t3, t3, 0x8\n"
                       "ble t0, %[numIndices], 1b\n"

                       "frcp.ps f29, f29 \n"
                       "fcvt.ps.pw f30, f30 \n"
                       "fmadd.ps f0, f0, f29, f30 \n"
                       "fcvt.pw.ps f0, f0 \n"
                       "fsat8.pi f0, f0 \n"
                       "fscb.ps  f0, f31(%[dst]) \n"

           

                       :
                       : [ gatherValues ] "r"(gatherValues),
                         [ src ] "r"(src),
                         [ numIndices ] "r"(numIndices),
                         [ batch ] "r"(batch),
                         [ tIndex ] "r"(tIndex),
                         [ batchPitch ] "r"(batchPitch),
                         [ typeSize ] "r"(typeSize),
                         [ dst ] "r"(dst),    
                         [ offset ] "r"(offset),
                         [ scale ] "r"(scale)
                           
                       : "t0", "t1", "t2", "t3", "f0", "f1", "f29", "f30", "f31", "memory"); 

}


template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value 
&& !std::is_same<srcType, float16>::value 
&& !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void sparseToDenseOp (uintptr_t dst, uintptr_t src, long long* tIndex, unsigned int batchPitch, 
unsigned int batch, unsigned int numIndices, size_t typeSize, float *scale, int32_t *offset){
}
                       

template <typename srcType>
void dnn_lib::fwdLibSparseToDenseInstVectorized(
    void *dstT, void *dstDims, void *dstPitches, void *srcT, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, void *indicesT, void *indDims,
    void *indPitches, float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  long long *tIndex = (long long *)indicesT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;
  unsigned int *indIndex = (unsigned int *)indDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;
  unsigned int *indPitch = (unsigned int *)indPitches;

  uintptr_t dstAddr = (uintptr_t)dstT;
  uintptr_t srcAddr = (uintptr_t)srcT;


  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  for (unsigned int i = 0; i < srcDimNum; i++)
    coord[i] = 0;
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetIn = 0;                              
  unsigned int offsetOut = 0;

  unsigned int batchPitch = srcPitch[0];
  srcPitch[0] = 0;

  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
    offsetIn += srcPitch[j] * coord[j];
  }
  
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;
  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  unsigned int elementsInRow, registersInRow, res;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

  while (!done && (offsetOut < posMax)) { 
    if (firstRow && coord[lastDim - 1] != maxRow) {
      elementsInRow = dstIndex[lastDim] - coord[lastDim];
    } else if (coord[lastDim - 1] == maxRow) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
    } else {
      elementsInRow = dstIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      res = elementsInRow - registersInRow * 8;
      mask = ((1 << res) - 1);
      __asm__ __volatile__("mov.m.x m1, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr += offsetIn * typeSize; 
    dstAddr += offsetOut * typeSize;

    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");


    for (unsigned int i = 0; i < registersInRow; i++) {
      sparseToDenseOp <srcType>(dstAddr, srcAddr, tIndex, batchPitch, coord[0], indIndex[0], typeSize, scale, offset);                  
      srcAddr += 8 * typeSize;
      dstAddr += 8 * typeSize;
    }
    if(res > 0) {
      __asm__ __volatile__("maskand m0, m1, m0 \n");
      sparseToDenseOp <srcType>(dstAddr, srcAddr, tIndex, batchPitch, coord[0], indIndex[0], typeSize, scale, offset);
    }
    if (lastRow) 
      return;
    
    dstAddr = (uintptr_t)dstT;
    srcAddr = (uintptr_t)srcT;
    offsetIn -= coord[lastDim] * srcPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;

    done = getOffsets(lastDim , coord, offsetIn, offsetOut, dstIndex,
                      srcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}





template <typename srcType>
void dnn_lib::fwdLibSparseToDenseMaskInst(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pdefault,
    int pdefaultSize, void *pindices, void *plengths, unsigned int pLengthsSize,
    void *pmask, unsigned int pMaskSize, float *scale, int32_t *offset) {

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  const Addresser<srcType> tDefVInput(pdefault, scale[1], offset[1]);
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;
  long long *mask = (long long *)pmask;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;

  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;

  // First un-processed index-value pair.
  size_t posIn = 0;
  // Beginning of output block for first unprocessed batch.
  size_t byteoffsetOut = 0;
  // Lengths can be scalar, which means that all pairs belong to one batch.
  size_t numBatches = (pLengthsSize == 1) ? 1 : lengths[0];
  // Go to the next batch
  size_t advanceBatch = (pLengthsSize == 1) ? 0 : dstPitch[0];
  // Go to the next position inside batch (row, column..)
  size_t advanceInBatch = (pLengthsSize == 1) ? dstPitch[0] : dstPitch[1];

  // Position of idx in the mask
  size_t j = 0;
  uint64_t srcAddr, srcAddrUp, dstAddr;
  for (size_t batch = 0; batch < numBatches; batch++) {
    // Fill everything with maskSize copies of defaultValue.
    for (size_t i = 0; i < pMaskSize; i++) {
      srcAddr = 0;
      srcAddrUp = pdefaultSize;
      dstAddr = byteoffsetOut + advanceInBatch * i;
      auto val = tDefVInput[0];
      for (uint64_t addr = srcAddr, cnt = 0; addr < srcAddrUp; addr++, cnt++) {
        val = tDefVInput[addr];
        tOutput[dstAddr + cnt] = val;
      }
    }
    // Go through input pairs and find matches.
    for (size_t i = 0; i < lengths[batch]; i++, posIn++) {
      auto idx = indices[posIn];
      // Search the mask
      for (j = 0; j < pMaskSize; j++) {
        if (mask[j] == idx) {
          break;
        }
      }
      // Skip if ID is not present in the mask.
      if (j == pMaskSize)
        continue;

      srcAddr = posIn * advanceInBatch;
      srcAddrUp = (posIn + 1) * advanceInBatch;
      dstAddr = byteoffsetOut + advanceInBatch * j;
      auto val = tAInput[0];
      for (uint64_t addr = srcAddr, cnt = 0; addr < srcAddrUp; addr++, cnt++) {
        val = tAInput[addr];
        tOutput[dstAddr + cnt] = val;
      }
    }

    byteoffsetOut += advanceBatch;
  }
}


// Assumptions for the SparseToDenseMaskInst threaded version:
// (1) The pmask vector size (pMaskSize) has the same length as the second dimension of the output tensor.
// (2) The dimensions and pitches of the pdefault tensor are the ones of a batch of the data tensor.

template <typename srcType>
void dnn_lib::fwdLibSparseToDenseMaskInstThreaded(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, unsigned int pdataDimNum, void *pdefault,
    int pdefaultSize, void *pindices, void *plengths, unsigned int pLengthsSize,
    void *pmask, unsigned int pMaskSize, float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  Addresser<srcType> tOutput(pdst, scale[2], offset[2]);
  const Addresser<srcType> tAInput(pdata, scale[0], offset[0]);
  const Addresser<srcType> tDefVInput(pdefault, scale[1], offset[1]);
  long long *indices = (long long *)pindices;
  int32_t *lengths = (int32_t *)plengths;
  long long *mask = (long long *)pmask;

  unsigned int *dstIndex = (unsigned int *)pdstDims;
  unsigned int *dataIndex = (unsigned int *)pdataDims;
  unsigned int *dstPitch = (unsigned int *)pdstPitches;
  unsigned int *dataPitch = (unsigned int *)pdataPitches;

  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];  
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions); 
  if (maxRead == 0)
    return;

  unsigned int coordOut[pdstDimNum];
  unsigned int last_non_zero_coord;
  getNonPaddingCoordinates(coordOut, initialAddr, pdstDimNum, dstPitch, dstIndex, last_non_zero_coord);

  uint64_t offsetOut = 0;
  for (unsigned int i = 0; i < last_non_zero_coord; i++) {
    offsetOut += dstPitch[i]*coordOut[i];
  }
  
  unsigned int batchCount = offsetOut/dstPitch[0];
  unsigned int mod = offsetOut - batchCount*dstPitch[0];
  unsigned int semiBatchCount = mod/dstPitch[1];
  unsigned int offsetIn = mod - semiBatchCount*dstPitch[1];

// The default value tensor's indexes and pitches can be obtained from the data tensor, as a consequence
// of assumption (2) listed above.
  unsigned int pdefDimNum = pdataDimNum - 1;
  unsigned int defPitch[pdefDimNum];
  unsigned int defIndex[pdefDimNum];
  for (int i = 0; i < pdefDimNum; i++) {
    defPitch[i] = dataPitch[i + 1];
    defIndex[i] = dataIndex[i + 1];
  }

  unsigned int coordIn[pdefDimNum]; // Coordinates in the default value tensor (or an input batch).
  getNonPaddingCoordinates(coordIn, offsetIn, pdefDimNum, defPitch, defIndex, last_non_zero_coord);
  offsetIn = 0;
  for (unsigned int i = 0; i < last_non_zero_coord; i++) {
    offsetIn += defPitch[i]*coordIn[i];
  }

  unsigned int firstIdx = 0;
  for (int i = 0; i < batchCount; ++i) firstIdx += lengths[i];
  unsigned int lastIdx = firstIdx + lengths[batchCount];

  unsigned int idx = mask[semiBatchCount];
  bool defaultVal = true;
  unsigned int j;

  for (j = firstIdx; j < lastIdx; j++) {
    if (indices[j] == idx) {
      defaultVal = false;
      break;
    }
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  bool doneIn = false;

  while (!done && (offsetOut < posMax)) {

    if (defaultVal) tOutput[offsetOut] = tDefVInput[offsetIn];
    else tOutput[offsetOut] = tAInput[j*dataPitch[0] + offsetIn];

    done = getOffsets(pdstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    doneIn = getOffsets(pdefDimNum, coordIn, offsetIn, defIndex, defPitch);

    if (doneIn) {
      doneIn = false;
      offsetIn = 0;
      for (unsigned int i = 0; i < pdefDimNum; ++i) {
        coordIn[i] = 0;
      }
      ++semiBatchCount;
      if (semiBatchCount == pMaskSize) { // Assumption (1): pMaskSize = dstIndex[1].
        semiBatchCount = 0;
        ++batchCount;
        firstIdx = lastIdx;
        lastIdx += lengths[batchCount];
      }
      idx = mask[semiBatchCount];
      defaultVal = true;
      for (j = firstIdx; j < lastIdx; j++) {
        if (indices[j] == idx) {
          defaultVal = false;
          break;
        }
      }

    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibLengthsToRangesInst(void *dstT, void *dstDims,
                                        void *dstPitches, void *plengths,
                                        unsigned int lenDim, float *scale,
                                        int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> lengths(plengths, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;

  auto toffset = lengths[0];
  toffset = 0;
  for (size_t i = 0; i < lenDim; i++) {
    auto length = lengths[i];
    tOutput[i * dstPitch[0]] = toffset;
    tOutput[i * dstPitch[0] + 1] = length;
    toffset += length;
  }
}

template <typename srcType>
void dnn_lib::fwdLibLengthsToRangesInstThreaded(void *dstT, void *dstDims,
                                        void *dstPitches, void *plengths,
                                        unsigned int lenDim, float *scale,
                                        int32_t *offset, uint64_t flags) {

  const Addresser<srcType> lengths(plengths, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[1], offset[1]);

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32;
  if (minionId >= activeMinions) return;
  int level = -1;
  for (int j = 1; j < activeMinions; j*=2)
    level++;    
  
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getReversedCachelinePartition(typeSize, lenDim, initialAddr, maxRead,
                               activeMinions);
 
  unsigned int posMax = maxRead + initialAddr;

  float toffset = lengths[0];
  toffset = 0;
  for (size_t i = initialAddr; i < posMax; i++)
    toffset += lengths[i];

#define tensor_reduce_params(_lvl)                    \
        ((0ULL  & 0x2)        << 62) |                \
        ((0ULL  & 0x1F)       << 57) |                \
        ((0ULL  & 0x1FFFFFFF) << 28) |                \
        ((0ULL  & 0xF)        << 24) |                \
        ((1ULL  & 0xFF)       << 16) |                \
        ((_lvl  & 0x1FFF)     << 3)  |                \
        ((0ULL  & 0x1)        << 2)  |                \
        ((0x3   & 0x3)) 
                               
#define tensor_broadcast_params(_lvl)                 \
        ((0ULL  & 0x2)        << 62) |                \
        ((0ULL  & 0x1F)       << 57) |                \
        ((0ULL  & 0x1FFFFFFF) << 28) |                \
        ((0ULL  & 0xF)        << 24) |                \
        ((1ULL  & 0xFF)       << 16) |                \
        ((_lvl  & 0x1FFF)     << 3)  |                \
        ((0ULL  & 0x1)        << 2)  |                \
        ((0x2   & 0x3))                              

#define REDUCE_AND_COPY(_J, _FD)                      \
        "csrw 0x800, %[csr_red" #_J "]\n"             \
        "fand.pi f" #_FD ", f0, f0\n"          

#define ADD_AND_BROADCAST(_J, _FD)                    \
        "fsub.ps f0, f0, f" #_FD "\n"                 \
        "csrw tensor_reduce, %[csr_bdc" #_J "]\n"     \
        "fadd.ps f0, f0, f" #_FD "\n"                 

  __asm__ __volatile__(
       "mov.m.x m0, zero, 0x1\n"
       "fmv.s.x f0, %[toffset]\n"
       "fand.pi f31, f0, f0\n"
      
       REDUCE_AND_COPY(0, 1)
       REDUCE_AND_COPY(1, 2)
       REDUCE_AND_COPY(2, 3)
       REDUCE_AND_COPY(3, 4)
       REDUCE_AND_COPY(4, 5)

       ADD_AND_BROADCAST(3, 4)
       ADD_AND_BROADCAST(2, 3)
       ADD_AND_BROADCAST(1, 2)
       ADD_AND_BROADCAST(0, 1)

       "fsub.ps f0, f0, f31\n" 
       "fmv.x.s %[toffset], f0\n"

     : [toffset] "+r" (toffset)
     : [csr_red0] "r" (tensor_reduce_params(0)),
       [csr_red1] "r" (tensor_reduce_params(1)),
       [csr_red2] "r" (tensor_reduce_params(2)),
       [csr_red3] "r" (tensor_reduce_params(3)),
       [csr_red4] "r" (tensor_reduce_params(4)),
       [csr_bdc0] "r" (tensor_broadcast_params(0)),
       [csr_bdc1] "r" (tensor_broadcast_params(1)),
       [csr_bdc2] "r" (tensor_broadcast_params(2)),
       [csr_bdc3] "r" (tensor_broadcast_params(3))
     : "f0", "f1", "f2", "f3", "f4", "f5", "f31"
   );

  for (size_t i = initialAddr; i < posMax; i++) {
    float length = lengths[i];
    tOutput[i * dstPitch[0]] = toffset;
    tOutput[i * dstPitch[0] + 1] = length;
    toffset += length;
  }

#undef tensor_reduce_params
#undef tensor_broadcast_params
#undef COPY_AND_REDUCE
#undef ADD_AND_BROADCAST
}

//===----------------------------------------------------------------------===//
//                Instructions used by RNN
//===----------------------------------------------------------------------===//

void swap(void *vals, void *inds, int i, int j) {
  float *fVals = (float *)vals;
  long long *lInds = (long long *)inds;

  float tval = fVals[i];
  long long tind = lInds[i];
  fVals[i] = fVals[j];
  lInds[i] = lInds[j];
  fVals[j] = tval;
  lInds[j] = tind;
}

int partition(void *vals, void *inds, int low, int high) {
  float *fVals = (float *)vals;
  long long *lInds = (long long *)inds;

  float pivotVal = fVals[high];
  long long pivotInd = lInds[high];
  int i = low - 1;

  for (int j = low; j <= high - 1; j++) {
    if (fVals[j] != pivotVal) {
      if (fVals[j] > pivotVal) {
        i++;
        swap(vals, inds, i, j);
      }
    } else if (lInds[j] < pivotInd) {
      i++;
      swap(vals, inds, i, j);
    }
  }
  swap(vals, inds, i + 1, high);
  return (i + 1);
}

void partialQuicksort(void *vals, void *inds, int low, int high, int m) {
  if (low < high) {
    int pidx = partition(vals, inds, low, high);
    partialQuicksort(vals, inds, low, pidx - 1, m);
    if (pidx < m) {
      partialQuicksort(vals, inds, pidx + 1, high, m);
    }
  }
}

// In this implementation we suppose that the dstPitches (1 and 2) have padding
// which ensures the dstPitches[n-2] being multiple of cacheline length if not,
// it needs sore global or reduce
template <typename srcType>
void dnn_lib::fwdLibTopKInst(void *dstT, void *dstDims, void *dstPitches,
                             void *dstT2, void *dst2Dims, void *dst2Pitches,
                             void *srcT, void *srcDims, void *srcPitches,
                             unsigned int srcDimNum, unsigned int k,
                             float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> inputT(srcT, scale[0], offset[0]);
  Addresser<srcType> valuesT(dstT, scale[1], offset[1]);

  long long *indT = (long long *)dstT2;

  unsigned int *valuesIndex = (unsigned int *)dstDims;
  unsigned int *indIndex = (unsigned int *)dst2Dims;
  unsigned int *inputIndex = (unsigned int *)srcDims;

  unsigned int *valuesPitch = (unsigned int *)dstPitches;
  unsigned int *indPitch = (unsigned int *)dst2Pitches;
  unsigned int *inputPitch = (unsigned int *)srcPitches;

  size_t n = inputIndex[srcDimNum - 1];
  // todo make it dependent of the type
  float tmpValues[n];
  long long tmpInd[n];

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eValuesPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eIndPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eInputPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum - 1; i++) {
    eDims[i] = inputIndex[i];
    eValuesPitch[i] = valuesPitch[i];
    eIndPitch[i] = indPitch[i];
    eInputPitch[i] = inputPitch[i];
  }

  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              // Do once per the most inner dimension
              for (size_t i = 0; i < n; i++) {
                tmpValues[i] = inputT[i * inputPitch[srcDimNum - 1] +
                                      x * eInputPitch[0] + y * eInputPitch[1] +
                                      z * eInputPitch[2] + w * eInputPitch[3] +
                                      q * eInputPitch[4] + r * eInputPitch[5]];
                tmpInd[i] = i;
              }
              partialQuicksort(&tmpValues[0], &tmpInd[0], 0, n - 1, k);
              for (size_t i = 0; i < k; i++) {
                valuesT[i * valuesPitch[srcDimNum - 1] + x * eValuesPitch[0] +
                        y * eValuesPitch[1] + z * eValuesPitch[2] +
                        w * eValuesPitch[3] + q * eValuesPitch[4] +
                        r * eValuesPitch[5]] = tmpValues[i];
                indT[i * indPitch[srcDimNum - 1] + x * eIndPitch[0] +
                     y * eIndPitch[1] + z * eIndPitch[2] + w * eIndPitch[3] +
                     q * eIndPitch[4] + r * eIndPitch[5]] = tmpInd[i];
              }
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibTopKInstThreaded_all(
    void *dstT, void *dstDims, void *dstPitches, void *dstT2, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, unsigned int k, float *scale, int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> inputT(srcT, scale[0], offset[0]);
  Addresser<srcType> valuesT(dstT, scale[1], offset[1]);

  long long *indT = (long long *)dstT2;

  unsigned int *valuesIndex = (unsigned int *)dstDims;
  unsigned int *indIndex = (unsigned int *)dst2Dims;
  unsigned int *inputIndex = (unsigned int *)srcDims;

  unsigned int *valuesPitch = (unsigned int *)dstPitches;
  unsigned int *indPitch = (unsigned int *)dst2Pitches;
  unsigned int *inputPitch = (unsigned int *)srcPitches;

  unsigned int numElemsValues = valuesPitch[0] * valuesIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsValues, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  for (int i = 0; i < srcDimNum; i++)
    coord[i] = 0;
  unsigned int l = 0;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, valuesPitch,
                           valuesIndex, l);

  unsigned int offsetValues = 0;
  unsigned int offsetInd = 0;
  for (size_t i = 0; i < l; i++) {
    offsetValues += coord[i] * valuesPitch[i];
    offsetInd += coord[i] * indPitch[i];
  }
  if (offsetValues >= numElemsValues)
    return;

  size_t n = inputIndex[srcDimNum - 1];
  // todo make it dependent of the type
  float tmpValues[n];
  long long tmpInd[n];

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  bool computed_topk = false;
  while (!done && offsetValues < posMax) {
    if (!computed_topk) {
      unsigned int offsetInput = 0;
      for (size_t j = 0; j < srcDimNum - 1; j++)
        offsetInput += coord[j] * inputPitch[j];
      for (size_t i = 0; i < n; i++) {
        tmpValues[i] = inputT[offsetInput + i * inputPitch[srcDimNum - 1]];
        tmpInd[i] = i;
      }
    }
    partialQuicksort(&tmpValues[0], &tmpInd[0], 0, n - 1, k);
    computed_topk = true;

    size_t i = coord[srcDimNum - 1];
    valuesT[offsetValues] = tmpValues[i];
    indT[offsetInd] = tmpInd[i];
    done = getOffsets(srcDimNum, coord, offsetValues, offsetInd, valuesIndex,
                      valuesPitch, indPitch);
    if (coord[srcDimNum - 1] == 0)
      computed_topk = false;
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

/*

template <typename srcType>
void dnn_lib::fwdLibTopKInstThreaded_k4(void *dstT, void *dstDims, void
*dstPitches, void *dstT2, void *dst2Dims, void *dst2Pitches, void *srcT, void
*srcDims, void *srcPitches, unsigned int srcDimNum, unsigned int k, float
*scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  __asm__ __volatile__ ("mov.m.x m0, zero, 0xff \n");

  const Addresser<srcType> inputT(srcT, scale[0], offset[0]);
  Addresser<srcType> valuesT(dstT, scale[1], offset[1]);

  long long *indT = (long long *)dstT2;
  srcType *valT = (srcType *)dstT;

  unsigned int *valuesIndex = (unsigned int *)dstDims;
  unsigned int *indIndex = (unsigned int *)dst2Dims;
  unsigned int *inputIndex = (unsigned int *)srcDims;

  unsigned int *valuesPitch = (unsigned int *)dstPitches;
  unsigned int *indPitch = (unsigned int *)dst2Pitches;
  unsigned int *inputPitch = (unsigned int *)srcPitches;

  unsigned int row_length = inputIndex[srcDimNum-1];
  unsigned int rows = 1;
  for (size_t i = 0; i < srcDimNum-1; i++) rows *= inputIndex[i];
  unsigned int max_minionsperrow = std::min(activeMinions/rows, row_length/8);

  unsigned int minionsperrow = 1;
  int level = -1;
  while (minionsperrow*2 <= max_minionsperrow) {
    minionsperrow *= 2;
    level++;
  }

  unsigned int row_id = minionId/minionsperrow;
  if (row_id >= rows) return;
  unsigned int row_minionId = minionId - row_id*minionsperrow;

  size_t n = 1;
  unsigned int current_length = 8 * minionsperrow;
  while (current_length < row_length) {
    n++;
    current_length += 8 * minionsperrow;
  }

  unsigned int batch_offset = row_id;
  unsigned int row_offset = n * 8 * row_minionId;
  unsigned int real_elements = n * 8;
  if (row_offset + n * 8 > row_length) {
    real_elements = row_length - row_offset;
    if (real_elements < 0) real_elements = 0;
  }
  n = real_elements/8;
  size_t remainder = real_elements - n * 8;

  long long indices[] = {0, 1, 2, 3, 4, 5, 6, 7}
  volatile int32_t srcOffset[] = {}

  volatile int32_t infty[] = 0xff800000;
  __asm__ __volatile__(
     "fxor.pi f0, f0, f0, f0\n"
     "fxor.pi f1, f1, f1, f1\n"
     "fxor.pi f2, f2, f2, f2\n"
     "fxor.pi f3, f3, f3, f3\n"
     "fbc.ps f0, 0x0(%[infty])\n"
     "fbc.ps f1, 0x0(%[infty])\n"
     "fbc.ps f2, 0x0(%[infty])\n"
     "fbc.ps f3, 0x0(%[infty])\n"
     :
     : [infty] "r" (infty)
     :
  );

  //for (int i = 0; i < n; i++) {



    __asm__ __volatile__(
        "1:\n"
        "fltm.ps m0, f3, f12\n" ////////////////
        "fand.pi f3, f12,f12\n" //            //
        "fand.pi f7, f13,f13\n" //            //
        "fand.pi f11,f14,f14\n" ////////////////

        "fltm.ps m0, f2, f3 \n" ////////////////
        "fand.pi f12,f3 ,f3 \n" //            //
        "fand.pi f3, f2 ,f2 \n" //            //
        "fand.pi f2, f12,f12\n" //            //
        "fand.pi f13,f7 ,f7 \n" //            //
        "fand.pi f7, f6 ,f6 \n" //            //
        "fand.pi f6, f13,f13\n" //            //
        "fand.pi f14,f11,f11\n" //            //
        "fand.pi f11,f10,f10\n" //            //
        "fand.pi f10,f14,f14\n" ////////////////

        "fltm.ps m0, f1, f2 \n" ////////////////
        "fand.pi f12,f2 ,f2 \n" //            //
        "fand.pi f2, f1 ,f1 \n" //            //
        "fand.pi f1, f12,f12\n" //            //
        "fand.pi f13,f6 ,f6 \n" //            //
        "fand.pi f6, f5 ,f5 \n" //            //
        "fand.pi f5, f13,f13\n" //            //
        "fand.pi f14,f10,f10\n" //            //
        "fand.pi f10,f9 ,f9 \n" //            //
        "fand.pi f9 ,f14,f14\n" ////////////////

        "fltm.ps m0, f0, f1 \n" ////////////////
        "fand.pi f12,f1 ,f1 \n" //            //
        "fand.pi f1, f0 ,f0 \n" //            //
        "fand.pi f0, f12,f12\n" //            //
        "fand.pi f13,f5 ,f5 \n" //            //
        "fand.pi f5, f4 ,f4 \n" //            //
        "fand.pi f4, f13,f13\n" //            //
        "fand.pi f14,f9 ,f9 \n" //            //
        "fand.pi f9 ,f8 ,f8 \n" //            //
        "fand.pi f8 ,f14,f14\n" ////////////////

        "mov.m.x m0, zero, 0xff\n"

        "addi    %[i], 1       \n"

        "blt     %[i], %[n], 1b\n"

      : [i] "+r"(i)
      : [n] "r"(n)
      : "f12", "f13", "f14", "f31"
    )

  //}

  //IF REMAINDER != 0

*/

// ONLY FOR K = 1, 2, 3, 4
template <typename srcType>
void dnn_lib::fwdLibTopKInstThreaded_k4(
    void *dstT, void *dstDims, void *dstPitches, void *dstT2, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, unsigned int k, float *scale, int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

  const Addresser<srcType> inputT(srcT, scale[0], offset[0]);
  Addresser<srcType> valuesT(dstT, scale[1], offset[1]);

  long long *indT = (long long *)dstT2;
  // srcType *valT = (srcType *)dstT;

  unsigned int *valuesIndex = (unsigned int *)dstDims;
  unsigned int *indIndex = (unsigned int *)dst2Dims;
  unsigned int *inputIndex = (unsigned int *)srcDims;

  unsigned int *valuesPitch = (unsigned int *)dstPitches;
  unsigned int *indPitch = (unsigned int *)dst2Pitches;
  unsigned int *inputPitch = (unsigned int *)srcPitches;

  unsigned int row_length = inputIndex[srcDimNum - 1];
  unsigned int rows = 1;
  for (size_t i = 0; i < srcDimNum - 1; i++)
    rows *= inputIndex[i];
  unsigned int max_minionsperrow =
      std::min(activeMinions / rows, row_length / 4);

  unsigned int minionsperrow = 1;
  int level = -1;
  while (minionsperrow * 2 <= max_minionsperrow) {
    minionsperrow *= 2;
    level++;
  }

  unsigned int row_id = minionId / minionsperrow;
  if (row_id >= rows)
    return;
  unsigned int row_minionId = minionId - row_id * minionsperrow;

  size_t n = row_length / minionsperrow;
  size_t remainder = row_length - minionsperrow * n;
  unsigned int batch_offset = row_id;
  long long row_offset;
  if (row_minionId < remainder) {
    n++;
    row_offset = row_minionId * n;
  } else {
    row_offset = remainder * (n + 1) + (row_minionId - remainder) * n;
  }

  unsigned int batchDim = (srcDimNum > 1) ? (srcDimNum - 2) : 0;
  indT += batch_offset * indPitch[batchDim];
  // valT += batch_offset * valuesPitch[srcDimNum - 2];

  float minusInf = - std::numeric_limits<float>::infinity();
  float tmpValues[5] = {minusInf, minusInf, minusInf, minusInf, minusInf};
  long long tmpInd[5] = {-1, -1, -1, -1, -1};
  long long final_offset = row_offset + n;
  for (long long i = row_offset; i < final_offset; i++) {
    tmpValues[4] = inputT[batch_offset * inputPitch[batchDim] +
                          i * inputPitch[srcDimNum - 1]];
    tmpInd[4] = i;
    for (int j = 3; j >= 0; j--) {
      if (tmpValues[j + 1] > tmpValues[j])
        swap(&tmpValues[0], &tmpInd[0], j + 1, j);
      else if ((tmpValues[j + 1] == tmpValues[j]) &&
               (tmpInd[j + 1] < tmpInd[j]))
        swap(&tmpValues[0], &tmpInd[0], j + 1, j);
    }
  }

  volatile int32_t gather_values[] = {0, 4, 8, 12, 0, 4, 8, 12};
  volatile int32_t gather_indices[] = {0, 8, 16, 24, 4, 12, 20, 28};
  __asm__ __volatile__("flw.ps  f31, 0x0(%[gather_values])\n"
                       "fgw.ps f0, f31(%[tmpValues])\n"
                       "flw.ps  f31, 0x0(%[gather_indices])\n"
                       "fgw.ps f1, f31(%[tmpInd])\n"

                       :
                       : [ gather_values ] "r"(gather_values),
                         [ gather_indices ] "r"(gather_indices),
                         [ tmpValues ] "r"(tmpValues), [ tmpInd ] "r"(tmpInd)
                       : "f31", "memory");

  unsigned int pow = 1;
  for (int j = 0; j <= level; j++) {
    uint64_t parity = ((minionId / pow + 1) & 0x1) * 2;
    uint64_t csr_enc = ((0ULL & 0x2) << 62) |
                       ((parity & 0x1F) << 57) | // Starting register
                       ((0ULL & 0x1FFFFFFF) << 28) |
                       ((8ULL & 0xF) << 24) |  // operation: 0x8 = get
                       ((2ULL & 0xFF) << 16) | // Number of registers
                       ((j & 0x1FFF) << 3) |   // Tree depth
                       ((0ULL & 0x1) << 2) | ((0x3 & 0x3)); // Tensor_reduce

    __asm__ __volatile__("csrw 0x800, %[csr_enc]\n"
                         "mov.m.x m0, zero, 0x11\n" // Set mask to 00010001
                         "fxor.pi f30, f30, f30\n"  // Clear f30
                         "fle.ps f30, f2, f0\n"
                         "fcmov.ps f4, f30, f0, f2\n"
                         "fcmov.ps f5, f30, f1, f3\n"
                         "mov.m.x m0, zero, 0xFF\n"    // Set mask to 11111111
                         "fswizz.ps f30, f30, 0x0\n"   // 0x00 = 00000000
                         "fcmov.ps f31, f30, f2, f0\n" // Rotate values
                         "fswizz.ps f31, f31, 0x93\n"
                         "fcmov.ps f0, f30, f0, f31\n"
                         "fcmov.ps f2, f30, f31, f2\n"
                         "fcmov.ps f31, f30, f3, f1\n" // Rotate indices
                         "fswizz.ps f31, f31, 0x93\n"
                         "fcmov.ps f1, f30, f1, f31\n"
                         "fcmov.ps f3, f30, f31, f3\n"
                         "mov.m.x m0, zero, 0x22\n" // Set mask to 00100010
                         "fxor.pi f30, f30, f30\n"  // Clear f30
                         "fle.ps f30, f2, f0\n"
                         "fcmov.ps f4, f30, f0, f2\n"
                         "fcmov.ps f5, f30, f1, f3\n"
                         "mov.m.x m0, zero, 0xFF\n"    // Set mask to 11111111
                         "fswizz.ps f30, f30, 0x55\n"  // 0x55 = 01010101
                         "fcmov.ps f31, f30, f2, f0\n" // Rotate values
                         "fswizz.ps f31, f31, 0x93\n"
                         "fcmov.ps f0, f30, f0, f31\n"
                         "fcmov.ps f2, f30, f31, f2\n"
                         "fcmov.ps f31, f30, f3, f1\n" // Rotate indices
                         "fswizz.ps f31, f31, 0x93\n"
                         "fcmov.ps f1, f30, f1, f31\n"
                         "fcmov.ps f3, f30, f31, f3\n"
                         "mov.m.x m0, zero, 0x44\n" // Set mask to 01000100
                         "fxor.pi f30, f30, f30\n"  // Clear f30
                         "fle.ps f30, f2, f0\n"
                         "fcmov.ps f4, f30, f0, f2\n"
                         "fcmov.ps f5, f30, f1, f3\n"
                         "mov.m.x m0, zero, 0xFF\n"    // Set mask to 11111111
                         "fswizz.ps f30, f30, 0xAA\n"  // 0xAA = 10101010
                         "fcmov.ps f31, f30, f2, f0\n" // Rotate values
                         "fswizz.ps f31, f31, 0x93\n"
                         "fcmov.ps f0, f30, f0, f31\n"
                         "fcmov.ps f2, f30, f31, f2\n"
                         "fcmov.ps f31, f30, f3, f1\n" // Rotate indices
                         "fswizz.ps f31, f31, 0x93\n"
                         "fcmov.ps f1, f30, f1, f31\n"
                         "fcmov.ps f3, f30, f31, f3\n"
                         "mov.m.x m0, zero, 0x88\n" // Set mask to 10001000
                         "fxor.pi f30, f30, f30\n"  // Clear f30
                         "fle.ps f30, f2, f0\n"
                         "fcmov.ps f4, f30, f0, f2\n"
                         "fcmov.ps f5, f30, f1, f3\n"
                         "mov.m.x m0, zero, 0xFF\n"    // Set mask to 11111111
                         "fswizz.ps f30, f30, 0xFF\n"  // 0xFF = 11111111
                         "fcmov.ps f31, f30, f2, f0\n" // Rotate values
                         "fswizz.ps f31, f31, 0x93\n"
                         "fcmov.ps f0, f30, f0, f31\n"
                         "fcmov.ps f2, f30, f31, f2\n"
                         "fcmov.ps f31, f30, f3, f1\n" // Rotate indices
                         "fswizz.ps f31, f31, 0x93\n"
                         "fcmov.ps f1, f30, f1, f31\n"
                         "fcmov.ps f3, f30, f31, f3\n"
                         "for.pi f0, f4, f4\n"
                         "for.pi f1, f5, f5\n"

                         :
                         : [ csr_enc ] "r"(csr_enc)
                         : "f0", "f1", "f2", "f3", "f4", "f5", "f29", "f30",
                           "f31");
    pow *= 2;
  }

  if (row_minionId == 0) {
    volatile int32_t gather_coord[] = {0, 8, 16, 24, 4, 12, 20, 28};
    volatile float tmpT[4];
    __asm__ __volatile__("flw.ps  f31, 0x0(%[gather_coord])\n"
                         "fscw.ps f1, f31(%[indT])\n"
                         "mov.m.x m0, zero, 0x0f\n"
                         "fsw.ps f0, 0x0(%[tmpT])\n"
                         :
                         : [ tmpT ] "r"(tmpT), [ indT ] "r"(indT),
                           [ gather_coord ] "r"(gather_coord)
                         : "f0", "f1", "f31");
    for (int i = 0; i < k; i++)
      valuesT[batch_offset * valuesPitch[batchDim] + i] = tmpT[i];
  }
}

// ONLY FOR K = 5, 6, 7, 8
template <typename srcType>
void dnn_lib::fwdLibTopKInstThreaded_k8(
    void *dstT, void *dstDims, void *dstPitches, void *dstT2, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, unsigned int k, float *scale, int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> inputT(srcT, scale[0], offset[0]);
  Addresser<srcType> valuesT(dstT, scale[1], offset[1]);

  long long *indT = (long long *)dstT2;

  unsigned int *valuesIndex = (unsigned int *)dstDims;
  unsigned int *indIndex = (unsigned int *)dst2Dims;
  unsigned int *inputIndex = (unsigned int *)srcDims;

  unsigned int *valuesPitch = (unsigned int *)dstPitches;
  unsigned int *indPitch = (unsigned int *)dst2Pitches;
  unsigned int *inputPitch = (unsigned int *)srcPitches;

  unsigned int row_length = inputIndex[srcDimNum - 1];
  unsigned int rows = 1;
  for (size_t i = 0; i < srcDimNum - 1; i++)
    rows *= inputIndex[i];
  unsigned int max_minionsperrow =
      std::min(activeMinions / rows, row_length / k);

  unsigned int minionsperrow = 1;
  int level = -1;
  while (minionsperrow * 2 <= max_minionsperrow) {
    minionsperrow *= 2;
    level++;
  }

  unsigned int row_id = minionId / minionsperrow;
  if (row_id >= rows)
    return;
  unsigned int rowMinionId = minionId - row_id * minionsperrow;

  size_t n = row_length / minionsperrow;
  size_t remainder = row_length - minionsperrow * n;
  unsigned int batch_offset = row_id;
  long long row_offset;
  if (rowMinionId < remainder) {
    n++;
    row_offset = rowMinionId * n;
  } else {
    row_offset = rowMinionId * n + remainder;
  }

  unsigned int batchDim = (srcDimNum > 1) ? (srcDimNum - 2) : 0;
  indT += batch_offset * indPitch[batchDim];
  
  float minusInf = -std::numeric_limits<float>::infinity();
  float tmpValues[9] = {minusInf, minusInf, minusInf, minusInf, minusInf,
                        minusInf, minusInf, minusInf, minusInf};
  long long tmpInd[9] = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  long long final_offset = row_offset + n;
  for (long long i = row_offset; i < final_offset; i++) {
    tmpValues[8] = inputT[batch_offset * inputPitch[batchDim] + i];
    tmpInd[8] = i;
    for (int j = 7; j >= 0; j--) {
      if (tmpValues[j + 1] > tmpValues[j])
        swap(tmpValues, tmpInd, j + 1, j);
      else if ((tmpValues[j + 1] == tmpValues[j]) &&
               (tmpInd[j + 1] < tmpInd[j]))
        swap(tmpValues, tmpInd, j + 1, j);
    }
  }
  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

  int32_t gather_values[] = {0, 4, 8, 12, 0, 4, 8, 12};
  int32_t gather_indices[] = {0, 8, 16, 24, 4, 12, 20, 28};
  __asm__ __volatile__("flw.ps  f31, 0x0(%[gather_values])\n"
                       "fgw.ps f0, f31(%[tmpValues])\n"
                       "faddi.pi f31, f31, 0x10\n" // New gather = {16, 20, 24,
                                                   // 28, 16, 20, 24, 28}
                       "fgw.ps f1, f31(%[tmpValues])\n"
                       "flw.ps  f31, 0x0(%[gather_indices])\n"
                       "fgw.ps f2, f31(%[tmpInd])\n"
                       "faddi.pi f31, f31, 0x20\n" // New gather = {32, 40, 48,
                                                   // 56, 36, 44, 52, 60}
                       "fgw.ps f3, f31(%[tmpInd])\n"

                       :
                       : [ gather_values ] "r"(gather_values),
                         [ gather_indices ] "r"(gather_indices),
                         [ tmpValues ] "r"(tmpValues), [ tmpInd ] "r"(tmpInd)
                       : "f31", "memory");

  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

#define TopKIteration1(_mask, _swizz)     \
       "mov.m.x m0, zero," #_mask "\n"    \
       "fxor.pi f30, f30, f30\n"          \
       "fle.ps f30, f4, f0\n"             \
       "fcmov.ps f8, f30, f0, f4\n"       \
       "fcmov.ps f10, f30, f2, f6\n"      \
                                          \
       "mov.m.x m0, zero, 0xFF\n"         \
       "fswizz.ps f30, f30," #_swizz "\n" \
                                          \
       "fcmov.ps f12, f30, f4, f0\n"      \
       "fcmov.ps f13, f30, f5, f1\n"      \
       "fswizz.ps f12, f12, 0x93\n"       \
       "fswizz.ps f13, f13, 0x93\n"       \
       "mov.m.x m0, zero, 0x11\n"         \
       "for.pi f13, f12, f12\n"           \
       "mov.m.x m0, zero, 0xFF\n"         \
       "fcmov.ps f0, f30, f0, f12\n"      \
       "fcmov.ps f4, f30, f12, f4\n"      \
       "fcmov.ps f1, f30, f1, f13\n"      \
       "fcmov.ps f5, f30, f13, f5\n"      \
                                          \
       "fcmov.ps f12, f30, f6, f2\n"      \
       "fcmov.ps f13, f30, f7, f3\n"      \
       "fswizz.ps f12, f12, 0x93\n"       \
       "fswizz.ps f13, f13, 0x93\n"       \
       "mov.m.x m0, zero, 0x11\n"         \
       "for.pi f13, f12, f12\n"           \
       "mov.m.x m0, zero, 0xFF\n"         \
       "fcmov.ps f2, f30, f2, f12\n"      \
       "fcmov.ps f6, f30, f12, f6\n"      \
       "fcmov.ps f3, f30, f3, f13\n"      \
       "fcmov.ps f7, f30, f13, f7\n"

#define TopKIteration2(_mask, _swizz)     \
       "mov.m.x m0, zero," #_mask "\n"    \
       "fxor.pi f30, f30, f30\n"          \
       "fle.ps f30, f5, f1\n"             \
       "fcmov.ps f9, f30, f1, f5\n"       \
       "fcmov.ps f11, f30, f3, f7\n"      \
                                          \
       "mov.m.x m0, zero, 0xFF\n"         \
       "fswizz.ps f30, f30," #_swizz "\n" \
                                          \
       "fcmov.ps f13, f30, f5, f1\n"      \
       "fswizz.ps f13, f13, 0x93\n"       \
       "fcmov.ps f1, f30, f1, f13\n"      \
       "fcmov.ps f5, f30, f13, f5\n"      \
                                          \
       "fcmov.ps f13, f30, f7, f3\n"      \
       "fswizz.ps f13, f13, 0x93\n"       \
       "fcmov.ps f3, f30, f3, f13\n"      \
       "fcmov.ps f7, f30, f13, f7\n"


  unsigned int pow = 1;
  for (int j = 0; j <= level; j++) {
    uint64_t startReg = ((minionId / pow + 1) & 0x1) * 4;
    uint64_t csr_enc = ((0ULL & 0x2) << 62)        |
                       ((startReg & 0x1F) << 57)   | // Starting register
                       ((0ULL & 0x1FFFFFFF) << 28) |
                       ((8ULL & 0xF) << 24)        |  // operation: 0x8 = get
                       ((4ULL & 0xFF) << 16)       | // Number of registers
                       ((j & 0x1FFF) << 3)         |   // Tree depth
                       ((0ULL & 0x1) << 2) | ((0x3 & 0x3)); // Tensor_reduce

    __asm__ __volatile__("csrw 0x800, %[csr_enc]\n"

                         TopKIteration1(0x11, 0x0)
                         TopKIteration1(0x22, 0x55)
                         TopKIteration1(0x44, 0xaa)
                         TopKIteration1(0x88, 0xff)
                         TopKIteration2(0x11, 0x0)
                         TopKIteration2(0x22, 0x55)
                         TopKIteration2(0x44, 0xaa)
                         TopKIteration2(0x88, 0xff)

                         "for.pi f0, f8, f8\n"
                         "for.pi f1, f9, f9\n"
                         "for.pi f2, f10, f10\n"
                         "for.pi f3, f11, f11\n"
                         :
                         : [ csr_enc ] "r"(csr_enc)
                         : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8",
                           "f9", "f10", "f11", "f12", "f13", "f30");
    pow *= 2;
  }

  if (rowMinionId == 0) {
    volatile int32_t gather_indices[] = {0, 8, 16, 24, 4, 12, 20, 28};
    volatile float tmpT[8];
    __asm__ __volatile__("flw.ps  f31, 0x0(%[gather_indices])\n"
                         "fscw.ps f2, f31(%[indT])\n"
                         "faddi.pi f31, f31, 0x20\n"
                         "fscw.ps f3, f31(%[indT])\n"
                         "mov.m.x m0, zero, 0x0F\n"
                         "fsw.ps f0, 0x0(%[tmpT])\n"
                         "fsw.ps f1, 0x10(%[tmpT])\n"

                         :
                         : [ tmpT ] "r"(tmpT), [ indT ] "r"(indT),
                           [ gather_indices ] "r"(gather_indices),
                           [ gather_values ] "r"(gather_values)
                         : "f0", "f1", "f2", "f3", "f31");
    for (int i = 0; i < k; i++)
      valuesT[batch_offset * valuesPitch[batchDim] + i] = tmpT[i];
  }
}

//===----------------------------------------------------------------------===//
//                Instructions used by Quantization
//===----------------------------------------------------------------------===//

/// Quantize floating point tensor. Scale and Offset are based on return type
/// of the instruction \p I.
template <typename dstType>
void dnn_lib::fwdLibQuantizeInst(void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, float scale,
                                 int32_t offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<dstType> ptrDstT(dstT, scale, offset);
  float *ptrSrcT = (float *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              int64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                z * eDstPitch[2] + w * eDstPitch[3] +
                                q * eDstPitch[4] + r * eDstPitch[5];
              int64_t srcAddr = x * eSrcPitch[0] + y * eSrcPitch[1] +
                                z * eSrcPitch[2] + w * eSrcPitch[3] +
                                q * eSrcPitch[4] + r * eDstPitch[5];
              auto val = ptrSrcT[srcAddr];
              // TODO check if we can use Addresser without breaking all the
              // other tests that uses int32_t as non quantized type
              if (std::is_same<dstType, int32_t>::value) {
                ptrDstT[dstAddr] = quantize<int32_t>(val, scale, offset);
              } else {
                ptrDstT[dstAddr] = val;
              }
            }
          }
        }
      }
    }
  }
}
template <typename dstType>
void dnn_lib::fwdLibQuantizeInstThreaded(void *dstT, void *dstDims,
                                         void *dstPitches, void *srcT,
                                         void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float scale,
                                         int32_t offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<dstType> ptrDstT(dstT, scale, offset);
  float *ptrSrcT = (float *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst =
      dstPitch[0] * srcIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position
  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k = 0;              // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, srcIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += srcPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done) {
    if (offsetOut >= posMax)
      break;
    auto val = ptrSrcT[offsetIn];
    // TODO check if we can use Addresser without breaking all the
    // other tests that uses int32_t as non quantized type
    if (std::is_same<dstType, int32_t>::value) {
      ptrDstT[offsetOut] = quantize<int32_t>(val, scale, offset);
    } else {
      ptrDstT[offsetOut] = val;
    }
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, srcIndex,
                      srcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

/// Dequantize integer tensor. Scale and Offset are based
/// on the source tensor type.
template <typename srcType>
void dnn_lib::fwdLibDequantizeInst(void *dstT, void *dstDims, void *dstPitches,
                                   void *srcT, void *srcDims, void *srcPitches,
                                   unsigned int srcDimNum, float scale,
                                   int32_t offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  float *ptrDstT = (float *)dstT;
  const Addresser<srcType> ptrSrcT(srcT, scale, offset);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              int64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                z * eDstPitch[2] + w * eDstPitch[3] +
                                q * eDstPitch[4] + r * eDstPitch[5];
              int64_t srcAddr = x * eSrcPitch[0] + y * eSrcPitch[1] +
                                z * eSrcPitch[2] + w * eSrcPitch[3] +
                                q * eSrcPitch[4] + r * eDstPitch[5];

              auto val = ptrSrcT[srcAddr];
              ptrDstT[dstAddr] = val;
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibDequantizeInstThreaded(void *dstT, void *dstDims,
                                           void *dstPitches, void *srcT,
                                           void *srcDims, void *srcPitches,
                                           unsigned int srcDimNum, float scale,
                                           int32_t offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  float *ptrDstT = (float *)dstT;
  const Addresser<srcType> ptrSrcT(srcT, scale, offset);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst =
      dstPitch[0] * srcIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;
  // We move the initialAddr to the next non-padding position
  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k = 0;              // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, srcIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += srcPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done) {
    ptrDstT[offsetOut] = ptrSrcT[offsetIn];
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, srcIndex,
                      srcPitch, dstPitch);
    if (offsetOut >= posMax)
      break;
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibRescaleQuantizedInst(void *dstT, void *dstDims,
                                         void *dstPitches, void *srcT,
                                         void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float srcScale,
                                         int32_t srcOffset, float dstScale,
                                         int32_t dstOffset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  srcType *ptrDstT = (srcType *)dstT;
  srcType *ptrSrcT = (srcType *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;
  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eDstPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              float val =
                  dequantize<srcType>(ptrSrcT[addrSrc], srcScale, srcOffset);
              ptrDstT[addrDst] = quantize<srcType>(val, dstScale, dstOffset);
            }
          }
        }
      }
    }
  }
}
 
template <typename srcType>
void dnn_lib::fwdLibRescaleQuantizedInstThreaded(
    void *dstT, void *dstDims, void *dstPitches, void *srcT, void *srcDims,
    void *srcPitches, unsigned int srcDimNum, float srcScale, int32_t srcOffset,
    float dstScale, int32_t dstOffset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  srcType *ptrDstT = (srcType *)dstT;
  srcType *ptrSrcT = (srcType *)srcT;

  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst =
      dstPitch[0] * srcIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);

  // We move the initialAddr to the next non-padding position
  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k = 0;              // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, srcIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += srcPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  uint64_t posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done) {
    float val = dequantize<srcType>(ptrSrcT[offsetIn], srcScale, srcOffset);
    ptrDstT[offsetOut] = quantize<srcType>(val, dstScale, dstOffset);
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, srcIndex,
                      srcPitch, dstPitch);
    if (offsetOut >= posMax)
      break;
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

void dnn_lib::fwdLibIntLookupTableInstInt8QTy(
    void *dstT, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src1T, void *src1Dims, void *src1Pitches, void *src2T, void *src2Dims,
    void *src2Pitches) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  int8_t *ptrDstT = (int8_t *)dstT;
  int8_t *ptrSrcT1 = (int8_t *)src1T;
  int8_t *ptrSrcT2 = (int8_t *)src2T;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *src1Index = (unsigned int *)src1Dims;
  unsigned int *src2Index = (unsigned int *)src2Dims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *src1Pitch = (unsigned int *)src1Pitches;
  unsigned int *src2Pitch = (unsigned int *)src2Pitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc1Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrc2Pitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < dstDimNum; i++) {
    eDims[i] = src1Index[i];
    eDstPitch[i] = dstPitch[i];
    eSrc1Pitch[i] = src1Pitch[i];
    eSrc2Pitch[i] = src2Pitch[i];
  }

  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              int val = ptrSrcT1[x * eSrc1Pitch[0] + y * eSrc1Pitch[1] +
                                 z * eSrc1Pitch[2] + w * eSrc1Pitch[3] +
                                 q * eSrc1Pitch[4] + r * eSrc1Pitch[5]];
              ptrDstT[x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                      w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5]] =
                  ptrSrcT2[val + 128];
            }
          }
        }
      }
    }
  }
}

void dnn_lib::fwdLibIntLookupTableInstInt8QTyThreaded(
    void *dstT, void *dstDims, void *dstPitches, unsigned int dstDimNum,
    void *src1T, void *src1Dims, void *src1Pitches, void *src2T, void *src2Dims,
    void *src2Pitches, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  int8_t *ptrDstT = (int8_t *)dstT;
  int8_t *ptrSrcT1 = (int8_t *)src1T;
  int8_t *ptrSrcT2 = (int8_t *)src2T;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *src1Index = (unsigned int *)src1Dims;
  unsigned int *src2Index = (unsigned int *)src2Dims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *src1Pitch = (unsigned int *)src1Pitches;
  unsigned int *src2Pitch = (unsigned int *)src2Pitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  getCachelinePartition(sizeof(int8_t), numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[dstDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex,
                           k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += src1Pitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;

  bool done = false;
  while (!done && (offsetOut < posMax)) {
    ptrDstT[offsetOut] = ptrSrcT2[ptrSrcT1[offsetIn] + 128];
    done = getOffsets(dstDimNum, coord, offsetIn, offsetOut, src1Index,
                      src1Pitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * sizeof(int8_t) / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + sizeof(int8_t)*initialAddr, clperminion);
}

template <typename srcType, typename dstType>
void dnn_lib::fwdLibConvertToInst(void *dstT, void *dstDims, void *dstPitches,
                                  void *srcT1, void *srcDims, void *srcPitches,
                                  unsigned int srcDimNum, float *scale,
                                  int32_t *offset) {

  // FIXME: single thread convertto fails when combined with multi-threaded
  // operators
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<dstType> ptrDstT(dstT, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT1(srcT1, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;
  Converter<srcType, dstType> converter;
  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              auto src = ptrSrcT1[addrSrc];
              if (std::is_same<srcType, dstType>::value) {
                ptrDstT[addrDst] = src;
              } else {
                auto dst = converter.convert(src);
                ptrDstT[addrDst] = dst;
              }
            }
          }
        }
      }
    }
  }
}

template <typename srcType, typename dstType>
void dnn_lib::fwdLibConvertToInstThreaded(void *dst, void *dstDims,
                                          void *dstPitches, void *src,
                                          void *srcDims, void *srcPitches,
                                          unsigned int srcDimNum, float *scale,
                                          int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<dstType> tOutput(dst, scale[1], offset[1]);
  const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  Converter<srcType, dstType> converter;

  unsigned int numElemsDst =
      dstPitch[0] * actIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position
  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k;                  // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    auto input = tAInput[offsetIn];
    auto output = converter.convert(input);
    tOutput[offsetOut] = output;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


template <typename srcType, typename dstType>
void dnn_lib::fwdLibConvertToInstVectorized(void *dst, void *dstDims,
                                            void *dstPitches, void *src,
                                            void *srcDims, void *srcPitches,
                                            unsigned int srcDimNum, float *scale,
                                            int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  //Addresser<dstType> tOutput(dst, scale[1], offset[1]);
  //const Addresser<srcType> tAInput(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  uintptr_t srcAddr = (uintptr_t)src;
  uintptr_t dstAddr = (uintptr_t)dst;

  Converter<srcType, dstType> converter;

  unsigned int numElemsDst =
      dstPitch[0] * dstIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSizeSrc = getsize<srcType>();
  size_t typeSizeDst = getsize<dstType>();
  getCachelinePartition(typeSizeDst, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (unlikely(maxRead == 0))
    return;

  // We move the initialAddr to the next non-padding position
  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k;                  // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, dstIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;

  unsigned int lastDim = srcDimNum - 1;

  volatile int32_t gatherValues[] = {0, 0, 0, 0, 0, 0, 0, 0};
  volatile int32_t scatterValues[] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (unsigned int i = 0; i < 8; i++) {
      gatherValues[i] = i * typeSizeSrc;
      scatterValues[i] = i * typeSizeDst;
  }

  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  unsigned int elementsInRow, registersInRow, res, spareElems, fullLanes;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  unsigned int elementsInRegister =  8 * (typeSizeDst != 8) + 4 * (typeSizeDst == 8);
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

  while (!done && (offsetOut < posMax)) {
    if (firstRow && coord[lastDim - 1] != maxRow) {
      elementsInRow = dstIndex[lastDim] - coord[lastDim];
    } else if (coord[lastDim - 1] == maxRow) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
    } else {
      elementsInRow = dstIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / elementsInRegister;
      res = elementsInRow - registersInRow * elementsInRegister;
      if (elementsInRegister != 4) {
        mask = ((1 << res) - 1);
      } else {
        mask = ((1 << 2*res) - 1);
      }
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr += offsetIn * typeSizeSrc; 
    dstAddr += offsetOut * typeSizeDst;
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

    unsigned int cnt = 0;
    while(cnt < registersInRow) {
      converter.convertVect(srcAddr, dstAddr, gatherValues, scatterValues); 
      cnt++;
      srcAddr += typeSizeSrc * elementsInRegister;
      dstAddr += typeSizeDst * elementsInRegister;
    }
    if (res > 0) {
      if (elementsInRegister != 4) {
        __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      } else {
        __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n"
                             "mov.m.x    m1, zero, 0x55 \n" 
                             "maskand m0, m0, m1 \n"
                             : : [ mask ] "r"(mask) :);
      }
      converter.convertVect(srcAddr, dstAddr, gatherValues, scatterValues); 
    }


    if (lastRow) 
      return;
    
    srcAddr = (uintptr_t)src;
    dstAddr = (uintptr_t)dst;
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;
    done = getOffsets(lastDim , coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSizeDst / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSizeDst*initialAddr, clperminion);
}

// This function copies a matrix replacing all the elements which are < splatVal
// and replaces them with splatVal
template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *srcT,
                                      void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, float splatVal,
                                      float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> ptrDstT(dstT, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT(srcT, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              auto src = ptrSrcT[addrSrc];
              ptrDstT[addrDst] = (src > splatVal) ? src : splatVal;
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *srcT,
                                      void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, int64_t splatVal,
                                      float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  srcType *ptrDstT = (srcType *)dstT;
  srcType *ptrSrcT = (srcType *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  uint64_t addrSrc, addrDst;

  // We can use this loop for all shapes.
  for (size_t x = 0; x < eBatchDims[0]; x++) {
    for (size_t y = 0; y < eBatchDims[1]; y++) {
      for (size_t z = 0; z < eBatchDims[2]; z++) {
        for (size_t w = 0; w < eBatchDims[3]; w++) {
          for (size_t q = 0; q < eBatchDims[4]; q++) {
            for (size_t r = 0; r < eBatchDims[5]; r++) {
              addrSrc = x * eSrcPitch[0] + y * eSrcPitch[1] + z * eSrcPitch[2] +
                        w * eSrcPitch[3] + q * eSrcPitch[4] + r * eSrcPitch[5];
              addrDst = x * eDstPitch[0] + y * eDstPitch[1] + z * eDstPitch[2] +
                        w * eDstPitch[3] + q * eDstPitch[4] + r * eDstPitch[5];
              int64_t src = ptrSrcT[addrSrc];
              ptrDstT[addrDst] = (src > splatVal) ? src : splatVal;
            }
          }
        }
      }
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInstThreaded(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              float splatVal, float *scale,
                                              int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> ptrDstT(dst, scale[1], offset[1]);
  const Addresser<srcType> ptrSrcT(src, scale[0], offset[0]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst =
      dstPitch[0] * actIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position

  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k = 0;              // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    float src = ptrSrcT[offsetIn];
    ptrDstT[offsetOut] = (src > splatVal) ? src : splatVal;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInstThreaded(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              int64_t splatVal, float *scale,
                                              int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  srcType *ptrDstT = (srcType *)dst;
  srcType *ptrSrcT = (srcType *)src;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst =
      dstPitch[0] * actIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position
  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k;                  // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    int64_t src = ptrSrcT[offsetIn];
    ptrDstT[offsetOut] = (src > splatVal) ? src : splatVal;
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value, 
std::size_t>::type = 0>
void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset){
  volatile int32_t gatherValues[] = {0, 4, 8, 12, 16, 20, 24, 28}; 
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                       "fgw.ps f0, f31(%[src])\n"
                       "fbc.ps f1, 0x0(%[splatVal])\n" 
                       "fmax.ps f0, f0, f1\n"
                       "fscw.ps  f0, f31(%[dst]) \n"

                       : 
                       : [ gatherValues ] "r"(gatherValues),
                         [ splatVal ] "r"(&splatVal),                        
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "f0", "f1", "f31", "memory"); 



}


template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, 
std::size_t>::type = 0>
void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset){
  volatile int32_t gatherValues[] = {0, 2, 4, 6, 8, 10, 12, 14}; 
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                       "fgh.ps f0, f31(%[src])\n"
                       "fcvt.ps.f16 f0, f0\n"
                       "fbc.ps f1, 0x0(%[splatVal])\n" 
                       "fmax.ps f0, f0, f1\n"
                       "fcvt.f16.ps f0, f0\n"  
                       "fsch.ps  f0, f31(%[dst]) \n"

                       : 
                       : [ gatherValues ] "r"(gatherValues),
                         [ splatVal ] "r"(&splatVal),                        
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "f0", "f1", "f31", "memory"); 
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value, 
std::size_t>::type = 0>
void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset ){
  
  volatile int32_t gatherValues[] = {0, 1, 2, 3, 4, 5, 6, 7}; 
  __asm__ __volatile__("flw.ps f31, 0x0(%[gatherValues])\n" 
                       "fgb.ps f0, f31(%[src])\n"
                       "fbc.ps f30, 0x0(%[offset]) \n"
                       "fbc.ps f29, 0x0(%[scale]) \n"
                       "fsub.pi f0, f0, f30 \n"
                       "fcvt.ps.pw f0, f0 \n"
                       "fmul.ps f0, f0, f29 \n"  
                       "fbc.ps f1, 0x0(%[splatVal])\n" 
                       "fmax.ps f0, f0, f1\n"
                       "frcp.ps f29, f29 \n"
                       "fcvt.ps.pw f30, f30 \n"
                       "fmadd.ps f0, f0, f29, f30 \n"
                       "fcvt.pw.ps f0, f0 \n"
                       "fsat8.pi f0, f0 \n"
                       "fscb.ps  f0, f31(%[dst]) \n"
                       : 
                       : [ gatherValues ] "r"(gatherValues),
                         [ splatVal ] "r"(&splatVal),                        
                         [ dst ] "r"(dst),
                         [ src ] "r"(src),
                         [ offset ] "r"(offset),
                         [ scale ] "r"(scale)

                       : "f0", "f1", "f29", "f30", "f31", "memory"); 
}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value 
&& !std::is_same<srcType, float16>::value 
&& !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void maxSplatOp (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset ){}

template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInstVectorized(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              float splatVal, float *scale,
                                              int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  uintptr_t dstAddr = (uintptr_t)dst;
  uintptr_t srcAddr = (uintptr_t)src;
  
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum]; 
  unsigned int k = 0;              
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;
  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  unsigned int elementsInRow, registersInRow, res;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false; 
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

  while (!done && (offsetOut < posMax)) {
  if (firstRow && coord[lastDim - 1] != maxRow) {
      elementsInRow = dstIndex[lastDim] - coord[lastDim];
    } else if (coord[lastDim - 1] == maxRow) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
    } else {
      elementsInRow = dstIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      res = elementsInRow - registersInRow * 8;
      mask = ((1 << res) - 1);
      __asm__ __volatile__("mov.m.x m1, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr += offsetIn * typeSize; 
    dstAddr += offsetOut * typeSize;

    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

    for (unsigned int i = 0; i < registersInRow; i++) {
      maxSplatOp <srcType>(dstAddr, srcAddr, splatVal, scale, offset);
      srcAddr += 8 * typeSize;
      dstAddr += 8 * typeSize;
    }

    if (res > 0) {
      __asm__ __volatile__("maskand m0, m1, m0 \n");
      maxSplatOp <srcType>(dstAddr, srcAddr, splatVal, scale, offset);
    }

    if (lastRow) 
      return;
    
    dstAddr = (uintptr_t)dst;
    srcAddr = (uintptr_t)src;
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;
 
  
  done = getOffsets(lastDim, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}


template <typename srcType, typename std::enable_if<std::is_same<srcType, float>::value, 
std::size_t>::type = 0>
void maxSplatOpAligned32Bytes (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset){
  __asm__ __volatile__("flw.ps f0, 0x0(%[src])\n" 
                       "fbc.ps f1, 0x0(%[splatVal])\n" 
                       "fmax.ps f0, f0, f1\n"
                       "fsw.ps  f0, 0x0(%[dst]) \n"
                       : 
                       : [ splatVal ] "r"(&splatVal),                        
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "f0", "f1", "f31", "memory"); 



}


template <typename srcType, typename std::enable_if<std::is_same<srcType, float16>::value, 
std::size_t>::type = 0>
void maxSplatOpAligned32Bytes (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset){
  __asm__ __volatile__( SET_FG32H_VAL(t0)
                       "fg32h.ps f0, t0(%[src])\n"
                       "fcvt.ps.f16 f0, f0\n"
                       "fbc.ps f1, 0x0(%[splatVal])\n" 
                       "fmax.ps f0, f0, f1\n"
                       "fcvt.f16.ps f0, f0\n"  
                       "fsc32h.ps f0, t0(%[dst]) \n"

                       : 
                       : [ splatVal ] "r"(&splatVal),                        
                         [ dst ] "r"(dst),
                         [ src ] "r"(src)
                       : "t0", "f0", "f1", "f31", "memory"); 
}

template <typename srcType, typename std::enable_if<std::is_same<srcType, int8_t>::value, 
std::size_t>::type = 0>
void maxSplatOpAligned32Bytes (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset ){
  
  __asm__ __volatile__(SET_FG32B_VAL(t0)
                       "fg32b.ps f0, t0(%[src])\n"
                       "fbc.ps f30, 0x0(%[offset]) \n"
                       "fbc.ps f29, 0x0(%[scale]) \n"
                       "fsub.pi f0, f0, f30 \n"
                       "fcvt.ps.pw f0, f0 \n"
                       "fmul.ps f0, f0, f29 \n"  
                       "fbc.ps f1, 0x0(%[splatVal])\n" 
                       "fmax.ps f0, f0, f1\n"
                       "frcp.ps f29, f29 \n"
                       "fcvt.ps.pw f30, f30 \n"
                       "fmadd.ps f0, f0, f29, f30 \n"
                       "fcvt.pw.ps f0, f0 \n"
                       "fsat8.pi f0, f0 \n"
                       "fsc32b.ps f0, t0(%[dst]) \n"
                       : 
                       : [ splatVal ] "r"(&splatVal),                        
                         [ dst ] "r"(dst),
                         [ src ] "r"(src),
                         [ offset ] "r"(offset),
                         [ scale ] "r"(scale)

                       : "t0", "f0", "f1", "f29", "f30", "f31", "memory"); 
}

template <typename srcType, typename std::enable_if<!std::is_same<srcType, int8_t>::value 
&& !std::is_same<srcType, float16>::value 
&& !std::is_same<srcType, float>::value, std::size_t>::type = 0>
void maxSplatOpAligned32Bytes (uintptr_t dst, uintptr_t src, float splatVal, float *scale, int32_t *offset ){}






template <typename srcType>
void dnn_lib::fwdLibETSOCMaxSplatInstAligned32Bytes(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              float splatVal, float *scale,
                                              int32_t *offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  uintptr_t dstAddr;
  uintptr_t srcAddr;
  
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *actIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *actPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum]; 
  unsigned int k = 0;              
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }


  unsigned int posMax = maxRead + initialAddr;
  bool done = false;

  unsigned int lastDim = srcDimNum - 1;
  unsigned int res = ((dstIndex[lastDim] - 1)%8) +1;
  actPitch[lastDim] *= 8;
  dstPitch[lastDim] *= 8;
  dstIndex[lastDim] = (dstIndex[lastDim] - 1)/8 + 1;
  unsigned int mask = ((1 << res) - 1);
  
  while (!done && (offsetOut < posMax)) {
    dstAddr = (uintptr_t)dst + offsetOut*typeSize;
    srcAddr = (uintptr_t)src + offsetIn*typeSize;

    if (coord[lastDim] != dstIndex[lastDim] - 1) 
         __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");
    else __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);

    maxSplatOpAligned32Bytes <srcType>(dstAddr, srcAddr, splatVal, scale, offset);

    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0)
    evict_va(0, DO_EVICTS, initialAddr, clperminion - 1, 64);
}





template <typename srcType>
void dnn_lib::fwdLibETSOCFullyConnectedInst(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  float *tBias = (float *)bias;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Add> opAdd;
  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Mul> opMul;
  uint64_t addrSrc1, addrSrc2, addrSrc3, addrDst;
  // For each (x,y) in the destination matrix:
  for (unsigned int x = 0; x < dstIndex[0]; x++) {
    for (unsigned int y = 0; y < dstIndex[1]; y++) {
      // Perform DOT on the row an column.
      float sum = 0.0;
      for (unsigned int i = 0; i < actIndex[1]; i++) {
        sum += float(tAInput[x * actPitch[0] + i] *
                     tWInput[i * weightPitch[0] + y]);
      }
      sum += tBias[y];
      tOutput[x * dstPitch[0] + y] = sum;
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibETSOCFullyConnectedInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  float *tBias = (float *)bias;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[2] = {0, 0};
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 2, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    float sum = 0;
    for (unsigned int i = 0; i < actIndex[1]; i++) {
      sum += float(tAInput[coord[0] * actPitch[0] + i]) *
             float(tWInput[i * weightPitch[0] + coord[1]]);
    }
    sum += tBias[coord[1]];
    tOutput[offsetOut] = float(sum);
    done = getOffsets(2, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, float>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

#define MATMUL_ITERATION               \
    "flw.ps   f0, 0x0(%[actAddr])\n"   \
    "fgw.ps   f1, f29(%[wgtAddr])\n"   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x20\n"
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"
                                         
    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f30, 0x0(%[biasAddr])\n"
    "fadd.s f31, f30, f31\n"
    "fsw.ps f31, 0x0(%[dstAddr])\n"
    
    :
    : [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr)
    : "t0", "t1", "f0", "f1", "f29", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, float16>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

#define MATMUL_ITERATION               \
    "fgh.ps   f0, f28(%[actAddr])\n"   \
    "fgh.ps   f1, f29(%[wgtAddr])\n"   \
    "fcvt.ps.f16 f0, f0 \n"            \
    "fcvt.ps.f16 f1, f1 \n"            \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f28, 0x0(%[gthValuesAct])\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x10\n" 
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"
                                         
    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f30, 0x0(%[biasAddr])\n"
    "fadd.s f31, f30, f31\n"
    "fcvt.f16.ps f31, f31\n"           // Conversion fp32 >> fp16.
    "fsch.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr)
    : "t0", "t1", "f0", "f1", "f28", "f29", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

#define INT8_TO_FP32(_reg)                  \
    "fsub.pi " #_reg ", " #_reg ", f26 \n"  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"    \
    "fmul.ps " #_reg ", " #_reg ", f27 \n"

#define MATMUL_ITERATION               \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f26, 0x0(%[offset]) \n"  \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                   \
    "fbc.ps   f26, 0x4(%[offset]) \n"  \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define FP32_TO_INT8(_reg)                      \
    "frcp.ps f27, f27 \n"                       \
    "fcvt.ps.pw f26, f26 \n"                    \
    "fmadd.ps " #_reg ", " #_reg ", f27, f26 \n" \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsat8.pi " #_reg ", " #_reg "\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f28, 0x0(%[gthValuesAct])\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x8\n"
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"
                                         
    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f30, 0x0(%[biasAddr])\n"
    "fadd.s f31, f30, f31\n"
    "fbc.ps f26, 0x8(%[offset]) \n"
    "fbc.ps f27, 0x8(%[scale]) \n"
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

#undef INT8_TO_FP32
#undef MATMUL_ITERATION
#undef FP32_TO_INT8
}

#define INT8_TO_FP32(_reg)                  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"    \
    "fmul.ps " #_reg ", " #_reg ", f27 \n"

#define UINT8_TO_FP32(_reg)                   \
    "fandi.pi " #_reg ", " #_reg ", 0xff \n"  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"      \
    "fmul.ps " #_reg ", " #_reg ", f27 \n"


#define MATMUL_ITERATION_U8_U8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    UINT8_TO_FP32(f0)                   \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    UINT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define MATMUL_ITERATION_I8_U8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                   \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    UINT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define MATMUL_ITERATION_U8_I8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    UINT8_TO_FP32(f0)                   \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define MATMUL_ITERATION_I8_I8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                   \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"


#define FP32_TO_INT8(_reg)                      \
    "frcp.ps f27, f27 \n"                       \
    "fmadd.ps " #_reg ", " #_reg ", f27, f26 \n" \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsat8.pi " #_reg ", " #_reg "\n"


#define FP32_TO_UINT8(_reg)                     \
    "frcp.ps f27, f27 \n"                       \
    "fmul.ps " #_reg ", " #_reg ", f27 \n"      \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsrli.pi f2," #_reg ", 0x8 \n"             \
    "fxor.pi f27, f27, f27 \n"                  \
    "fcmov.ps " #_reg" , f12, f27, " #_reg " \n"  


#define STEP1                                            \
    "mov.m.x m0, zero, 0xff\n"                           \
    "xor t0, t0, t0\n"                                   \
    "flw.ps f28, 0x0(%[gthValuesAct])\n"                 \
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"                 \
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"                   \
    "fxor.pi f31, f31, f31\n"                            \
                                                         \
    "1:\n"                                               \
    "addi     t0, t0, 8\n"                               \
    "ble      %[elemsRow], t0, 2f\n"

#define STEP2                                            \
    "addi %[actAddr], %[actAddr], 0x8\n"                 \
    "fadd.pi f29, f29, f30\n"                            \
    "beq      zero, zero, 1b\n"                          \
    "2:\n"                                               \
    "fxor.pi  f0, f0, f0\n"                              \
    "addi     t0, t0, -8\n"                              \
    "sub      t0, %[elemsRow], t0\n"                     \
    "addi     t1, zero, 1\n"                             \
    "sll      t1, t1, t0\n"                              \
    "addi     t1, t1, -1\n"                              \
    "mov.m.x  m0, t1, 0\n"                               \

#define STEP3                                            \
    "fmvs.x.ps t0, f31, 0x4\n"                           \
    "fmv.w.x   f30, t0\n"                                \
    "fadd.s    f31, f30, f31\n"                          \
    "mov.m.x m0, zero, 0x1\n"                            \
    "fbc.ps f30, 0x0(%[biasAddr])\n"                     \
    "fadd.s f31, f30, f31\n"                             \
    "fbc.ps f26, 0x8(%[offset]) \n"                      \
    "fbc.ps f27, 0x8(%[scale]) \n"                       \


template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_I8
    STEP2
    MATMUL_ITERATION_U8_I8
    STEP3
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){
  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_I8_U8
    STEP2
    MATMUL_ITERATION_I8_U8
    STEP3
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_I8_I8
    STEP2
    MATMUL_ITERATION_I8_I8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_I8_U8
    STEP2
    MATMUL_ITERATION_I8_U8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_I8
    STEP2
    MATMUL_ITERATION_U8_I8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");


}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset) {

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_U8
    STEP2
    MATMUL_ITERATION_U8_U8
    STEP3
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_U8
    STEP2
    MATMUL_ITERATION_U8_U8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"
    
    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}



template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<!std::is_same<src1Type, int8_t>::value && !std::is_same<src1Type, float16>::value && !std::is_same<src1Type, float>::value && !std::is_same<src1Type, uint8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){}

template <typename src1Type, typename src2Type, typename dstType>
void dnn_lib::fwdLibETSOCFullyConnectedInstVectorized(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches1,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches1;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<src1Type>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[2];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 2, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  unsigned int offsetAIn = coord[0]*actPitch[0];

  int32_t gatherValuesAct[8], gatherValuesWgt[8];
  gatherValuesAct[0] = gatherValuesWgt[0] = 0;
  unsigned int step = weightPitch[0]*typeSize;
  for (unsigned int i = 1; i < 8; ++i) {
    gatherValuesAct[i] = gatherValuesAct[i - 1] + typeSize;
    gatherValuesWgt[i] = gatherValuesWgt[i - 1] + step;
  }
  unsigned int wgtRegStep = 8*step;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    uintptr_t dstAddr = (uintptr_t)dstMatrix + typeSize*offsetOut;
    uintptr_t actAddr = (uintptr_t)activations + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)weights + typeSize*coord[1];
    uintptr_t biasAddr = (uintptr_t)bias + 4*coord[1]; // bias is a float vector.
    fullyConnectedOp <src1Type, src2Type, dstType>(dstAddr, actAddr, wgtAddr, actIndex[1], gatherValuesAct,
                       gatherValuesWgt, wgtRegStep, biasAddr, scale, offset);
    done = getOffsets(2, coord, offsetOut, dstIndex, dstPitch);
    if (coord[1] == 0) {
      offsetAIn += actPitch[0];
    } 
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

