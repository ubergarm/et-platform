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

#ifndef UTILS_H
#define UTILS_H

#include <sstream>
#include <cstdint>

#include <syscall.h>
#include <device_common.h>

#include "LibCommon.h"
#include "Float16.h"

using namespace dnn_lib;

#define MAX_TENSOR_DIMENSIONS 6

#define ACTIVE_SHIRES ((flags & 0x1F) + 1)
#define DO_EVICTS     ((flags & 0x60) >> 5) // 01 = evictL2, 10 = evictL3, 11 = evictMem

#define M_1_LOG2E float(1.0f / M_LOG2E)
#define SET_FG32B_VAL(_reg) "li " #_reg ", 0x398A418820\n"
#define SET_FG32H_VAL(_reg) "li " #_reg ", 0x76543210\n"
//   No SET_FG32W.PS_VAL(_reg) because flw.ps is used instead of fg32w.ps.
#define SET_MINUS_INFTY(_reg) "fbci.ps " #_reg ", 0xff800 \n" // _reg is vect

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

#define print(s) syscall(SYSCALL_LOG_WRITE, (uint64_t)(s), sizeof(s), 0)

//-------------------------------------------------------------------------------------------------
//
// FUNCTION: evict_va_multi
//
//   This function is a wrapper of evict_va for any number for cache lines. It calls evict_va as
//   many times as needed to evict all lines
//
inline __attribute__((always_inline))
void evict_va_multi(uint64_t dst, uintptr_t addr, uint64_t num_lines) {
  while (num_lines > 16) {
    evict_va(0, dst, addr, 15, 64);
    addr += 64;
    num_lines -= 16;
  }
  if (num_lines > 0)
    evict_va(0, dst, addr, num_lines-1, 64);
}

inline __attribute__((always_inline))
unsigned int gcd(unsigned int a, unsigned int b) {
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

inline __attribute__((always_inline))
void getCoordinates(unsigned int *coord, unsigned int address,
                    unsigned int srcDimNum, unsigned int *pitch) {
  unsigned int rm = address;
  for (unsigned int i = 0; i < srcDimNum; i++) {
    coord[i] = rm / pitch[i];
    rm = rm - coord[i] * pitch[i];
  }
}

inline __attribute__((always_inline))
void getNonPaddingCoordinates(unsigned int *coord, unsigned int address,
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
inline __attribute__((always_inline))
void getCachelinePartition(unsigned int elementsize, unsigned int numElemsDst,
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
inline __attribute__((always_inline))
void getUniformCachelinePartition(unsigned int elementsize, unsigned int numElemsDst,
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

inline __attribute__((always_inline))
void getReversedCachelinePartition(unsigned int elementsize, unsigned int numElemsDst,
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
inline __attribute__((always_inline))
bool getOffsets(unsigned int DimNum, unsigned int *coord, mytype &offsetAddr,
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
inline __attribute__((always_inline))
bool getOffsets(unsigned int DimNum, unsigned int *coord, mytype &offsetAddr1,
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
inline __attribute__((always_inline))
bool getOffsets(unsigned int DimNum, unsigned int *coord, mytype &offsetAddr1,
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
inline __attribute__((always_inline))
bool getOffsets(unsigned int DimNum, unsigned int *coord, mytype &offsetAddr1,
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

inline __attribute__((always_inline))
bool getOffsets(unsigned int dimNum, unsigned int *coord,
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

inline __attribute__((always_inline))
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

inline __attribute__((always_inline))
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

inline __attribute__((always_inline))
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

inline __attribute__((always_inline))
bool isInteger(float f) {
  return (nearbyintf(f) == f);
}

inline __attribute__((always_inline))
bool isEven(int val) {
  return ((val % 2) == 0);
}

inline
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

template <typename srcType>
inline __attribute__((always_inline))
void getLanesResTView (int &lanes, int &res, const unsigned int &d) {
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

inline __attribute__((always_inline))
bool getNextStep(unsigned int dimNum,
                 unsigned int *coord, unsigned int *dims) {
  if (coord[0] < dims[0]-1) {
    coord[0] = coord[0]+1;
  } else {
    coord[0] = 0;
    for (int i = 1; i < dimNum; i++) {
      if (coord[i] < dims[i]-1) {
        coord[i] = coord[i]+1;
        break;
      } else {
        coord[i] = 0;
        if (i == dimNum-1)
          return true;
      }
    }
  }
  return false;
}

inline __attribute__((always_inline))
unsigned int getOffset(unsigned int *coord,  unsigned int dimNum,
                       unsigned int *pitch) {
  unsigned int offset = 0;
  for (int i = 0; i < dimNum; i++) {
    offset += coord[i] * pitch[i];
  }
  return offset;
}

#endif /* UTILS_H */
