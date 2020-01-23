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
#include <utility>

#include <syscall.h>
#include <device_common.h>


#include "cacheops.h"
#include "LibCommon.h"
#include "Float16.h"

using namespace dnn_lib;

#define MAX_TENSOR_DIMENSIONS 6

#define ACTIVE_SHIRES ((flags & 0x1F) + 1)
#define DO_EVICTS     ((flags & 0x60) >> 5) // 01 = evictL2, 10 = evictL3, 11 = evictMem

#define M_1_LOG2E float(1.0f / M_LOG2E)

constexpr uint64_t fg32b_conf = 0x398A418820;
constexpr uint64_t fg32h_conf = 0x76543210;

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
    addr += (64*16);
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

/**
 * @brief Converts an offset in a tensor into its corresponding coordinates.
 *
 * This function takes into account the padding that the tensor my have, but if
 * the offset corresponds to a padding position, the returned coordinates will
 * point this padding position in the matrix (outside the dimensions).
 *
 * @param[out] coord Vector that will be filled with the coordinates of the 
 *  offset in the tensor.
 * @param[in] offset Unsigned integer referring to a position in a tensor.
 * @param[in] srcDimNum The "number of dimensions" of the tensor.
 * @param[in] pitch The vector of pitches of the given tensor.
 */
inline __attribute__((always_inline))
void getCoordinates(unsigned int *coord, unsigned int offset,
                    unsigned int dimNum, unsigned int *pitch) {
  unsigned int rm = offset;
  for (unsigned int i = 0; i < dimNum; i++) {
    coord[i] = rm / pitch[i];
    rm = rm - coord[i] * pitch[i];
  }
}




/**
 * @brief Converts an offset in a tensor into its corresponding non-padding-coords.
 *
 * This function takes into account the padding that the tensor my have, and if
 * the offset corresponds to a padding position, the returned coordinates will
 * point the next position in the tensor that doesn't correspond to padding.
 *
 * @param[out] coord Vector that will be filled with the coordinates of the 
 *  offset in the tensor.
 * @param[in] offset Unsigned integer referring to a position in a tensor.
 * @param[in] srcDimNum The "number of dimensions" of the tensor.
 * @param[in] pitch The vector of pitches of the given tensor.
 * @param[in] dims The sizes of each dimension of the given tensor.
 * @param[out] k The last coordinate that has not been set to 0 while searching
 *  the next non-padding-position.
 */
inline __attribute__((always_inline)) 
void getNonPaddingCoordinates(unsigned int *coord, unsigned int offset,
                              unsigned int srcDimNum, unsigned int *pitch,
                              unsigned int *dims, unsigned int &k) {

  getCoordinates(coord, offset, srcDimNum, pitch);
  k = srcDimNum;
  for (int j = srcDimNum - 1; j > 0; j--) {
    if (unlikely(coord[j] >= dims[j])) {
      coord[j - 1]++;
      k = j;
    }
  }
  for (unsigned int j = k; j < srcDimNum; j++)
    coord[j] = 0;
}

/**
 * @brief Given a tensor, it divides it in cachelines for the minions.
 *
 * It gives to each minion an offset to start and how many elements to work on. 
 * The division is made such that the there is no cacheline for two different minions. 
 * The division ensures that all active minions minions (except for, possibly, 
 * the last one) work with the same number of cachelines and the following
 * have no positions to work with.
 * 
 * @warning The number maxRead does not take into account padding, so if maxRead
 *  is 16, that does not mean that the minion has to work on 16 elements: some of
 *  them may be padding. Moreover, @f$ offset + maxRead@f$ may be outside the tensor.
 *
 * @warning The function works with the supposition that the minions working on this
 *  tensor is numbered from 0 to activeMinions.
 *
 * @warning It is assumed that the tensor that starts at cacheline.
 *
 * @param[in] elementSize The number of bytes of each element in the matrix.
 *  It is required to be a power of 2 and smaller than 64 (1, 2, 4, 8, usually).
 * @param[in] numElems The number of elements in the tensor that is divided.
 * @param[out] offset The starting offset for the minion.
 * @param[out] maxRead The number of consecutive elements the minion is assigned.
 * @param[in] minionId The id of the minion that calls the function.
 * @param[in] activeMinions The number of minions that is working on the tensor.
 */
inline __attribute__((always_inline)) 
void getCachelinePartition(unsigned int elementSize, unsigned int numElems,
                           unsigned int &offset, unsigned int &maxRead,
                           unsigned int minionId, unsigned int activeMinions) {

  unsigned int cll = 64 / elementSize;              // Cacheline length
  unsigned int ncl = (numElems - 1) / cll + 1;      // Amount of cache lines
  unsigned int mcl = (ncl - 1) / activeMinions + 1; // Amount of cl for a minion
  unsigned int div = ncl / mcl;

  if (minionId < div) {
    maxRead = mcl * cll;
    offset = maxRead * minionId;
  } else if (minionId == div) {
    maxRead = (ncl - div * mcl) * cll;
    offset = mcl * cll * minionId;
  } else
    maxRead = 0;
}

/**
 * @brief Given a tensor, it divides it in cachelines for the minions.
 *
 * It gives to each minion an offset to start and how many elements to work on. 
 * The division is made such that the there is no cacheline for two different minions. 
 * The division ensures that the amount of cachelines for all the working minions 
 * differs at most for one except for the ones that have no cachelines assigned.
 * For a pair of minions with a non zero number of cachelines assigned it is true that
 * the minion with a smaller id works on a greater or equal number of cachelines.
 * 
 * @warning The number maxRead does not take into account padding, so if maxRead
 *  is 16, that does not mean that the minion has to work on 16 elements: some of
 *  them may be padding. Moreover, @f$ offset + maxRead@f$ may be outside the tensor.
 *
 * @warning The function works with the supposition that the minions working on this
 *  tensor is numbered from 0 to activeMinions.
 *
 * @param[in] elementSize The number of bytes of each element in the matrix.
 *  It is required to be a power of 2 and smaller than 64 (1, 2, 4, 8, usually).
 * @param[in] numElems The number of elements in the tensor that is divided.
 * @param[out] offset The starting offset for the minion.
 * @param[out] maxRead The number of consecutive elements the minion is assigned.
 * @param[in] activeMinions The number of minions that is working on the tensor.
 */
inline __attribute__((always_inline))
void getUniformCachelinePartition(unsigned int elementsize, unsigned int numElems,
                                  unsigned int &offset, unsigned int &maxRead,
                                  unsigned int activeMinions) {

  unsigned int minionId = get_minion_id();
  unsigned int cll = 64 / elementsize;            // Cacheline lenght
  unsigned int ncl = (numElems - 1) / cll + 1;    // Amount of cache lines
  unsigned int mcl = ncl / activeMinions;         // Amount of cl for the minion
  unsigned int mod = ncl - activeMinions * mcl;
  if (minionId < mod) {
    ++mcl;
    offset = mcl * cll * minionId;
  }
  else {
    offset = (mod + minionId * mcl) * cll;
  }
  maxRead = mcl * cll;
}

/**
 * @brief Given a tensor, it divides it in cachelines for the minions.
 *
 * It gives to each minion an offset to start and how many elements to work on. 
 * The division is made such that the there is no cacheline for two different minions. 
 * The division ensures that the amount of cachelines for all the working minions 
 * differs at most for one except for the ones that have no cachelines assigned.
 * For a pair of minions with a non zero number of cachelines assigned it is true that
 * the minion with a greater id works on a greater or equal number of cachelines.
 * 
 * @warning The number maxRead does not take into account padding, so if maxRead
 *  is 16, that does not mean that the minion has to work on 16 elements: some of
 *  them may be padding. Moreover, @f$ offset + maxRead@f$ may be outside the tensor.
 *
 * @warning The function works with the supposition that the minions working on this
 *  tensor is numbered from 0 to activeMinions.
 *
 * @param[in] elementSize The number of bytes of each element in the matrix.
 *  It is required to be a power of 2 and smaller than 64 (1, 2, 4, 8, usually).
 * @param[in] numElems The number of elements in the tensor that is divided.
 * @param[out] offset The starting offset for the minion.
 * @param[out] maxRead The number of consecutive elements the minion is assigned.
 * @param[in] minionId The id of the minion that calls the function.
 * @param[in] activeMinions The number of minions that is working on the tensor.
 */
inline __attribute__((always_inline)) 
void getReversedCachelinePartition(unsigned int elementsize, unsigned int ElemsDst,
                                   unsigned int &offset, unsigned int &maxRead,
                                   unsigned int activeMinions) {
  unsigned int minionId = (activeMinions - get_minion_id()) - 1;
  unsigned int cll = 64 / elementsize;            // Cacheline length
  unsigned int ncl = (ElemsDst - 1) / cll + 1;    // Amount of cache lines
  unsigned int mcl = ncl / activeMinions;         // Amount of cl for a minion
  unsigned int mod = ncl - activeMinions * mcl;
  if (minionId < mod) {
    ++mcl;
    offset = mcl * cll * minionId;
  } else
    offset = (mod + minionId * mcl) * cll;

  maxRead = mcl * cll;
}

/**
 * @brief Updates a position in a tensor into the next non-padding position.
 *
 * Both the coordinates of the position and the offset are updated. It also
 * returns if the end of the tensor has been reached.
 * 
 * @tparam T The type of the elements in the tensor.
 * @param[in] dimNum The "number of dimensions" of the tensor.
 * @param[in, out] coord Vector of coordinates of the initial position.
 *  It is updated into the new coordinates.
 * @param[in, out] offset The starting offset. It is updated into the new offset.
 * @param[in] index The sizes of each dimension of the given tensor.
 * @param[in] pitch The vector of pitches of the given tensor.
 * @returns True if the tensor has ended and false otherwise.
 */
template <typename T>
inline __attribute__((always_inline)) 
bool getOffsets(unsigned int dimNum, unsigned int *coord, T &offset,
                unsigned int *index, unsigned int *pitch) {

  for (int j = dimNum - 1; j >= 0; j--) {
    if (likely(coord[j] != (index[j] - 1))) {
      offset += pitch[j];
      coord[j]++;
      return false;
    } else if (likely(j != 0)) {
      offset -= (index[j] - 1) * pitch[j];
      coord[j] = 0;
    } else
      return true;
  }

  //FIXME: use assertion throw "getOffsets Malfunction";
  // To avoid warnings. This point will never be reached.
  return true; 
}


/**
 * @brief Updates a position in two tensors into the next non-padding position.
 *
 * @overload
 * 
 * @warning The tensors in which we are moving should have the same dimensions
 *  (but not necessarily the same pitches).
 */
template <typename T>
inline __attribute__((always_inline)) 
bool getOffsets(unsigned int dimNum, unsigned int *coord, T &offset1,
                T &offset2, unsigned int *index, unsigned int *pitch1,
                unsigned int *pitch2) {

  for (int j = dimNum - 1; j >= 0; j--) {
    if (likely(coord[j] != (index[j] - 1))) {
      offset1 += pitch1[j];
      offset2 += pitch2[j];
      coord[j]++;
      return false;
    } else if (likely(j != 0)) {
      offset1 -= (index[j] - 1) * pitch1[j];
      offset2 -= (index[j] - 1) * pitch2[j];
      coord[j] = 0;
    } else
      return true;
  }

  //FIXME: use assertion throw "getOffsets Malfunction";
  // To avoid warnings. This point will never be reached.
  return true; 
}

/**
 * @brief Updates a position in three tensors into the next non-padding position.
 *
 * @overload
 * 
 * @warning The tensors in which we are moving should have the same dimensions
 *  (but not necessarily the same pitches).
 */
template <typename T> 
inline __attribute__((always_inline)) 
bool getOffsets(unsigned int dimNum, unsigned int *coord, T &offset1,
                T &offset2, T &offset3, unsigned int *index, unsigned int *pitch1,
                unsigned int *pitch2, unsigned int *pitch3) {

  for (int j = dimNum - 1; j >= 0; j--) {
    if (likely(coord[j] != (index[j] - 1))) {
      offset1 += pitch1[j];
      offset2 += pitch2[j];
      offset3 += pitch3[j];
      coord[j]++;
      return false;
    } else if (likely(j != 0)) {
      offset1 -= (index[j] - 1) * pitch1[j];
      offset2 -= (index[j] - 1) * pitch2[j];
      offset3 -= (index[j] - 1) * pitch3[j];
      coord[j] = 0;
    } else
      return true;
  }

  //FIXME: use assertion throw "getOffsets Malfunction";
  // To avoid warnings. This point will never be reached.
  return true;
}

/**
 * @brief Updates a position in four tensors into the next non-padding position.
 *
 * @overload
 * 
 * @warning The tensors in which we are moving should have the same dimensions
 *  (but not necessarily the same pitches).
 */
template <typename T> 
inline __attribute__((always_inline)) 
bool getOffsets(unsigned int dimNum, unsigned int *coord, T &offset1,
                T &offset2, T &offset3, T &offset4, unsigned int *index, 
                unsigned int *pitch1, unsigned int *pitch2,
                unsigned int *pitch3, unsigned int *pitch4) {

  for (int j = dimNum - 1; j >= 0; j--) {
    if (coord[j] != (index[j] - 1)) {
      offset1 += pitch1[j];
      offset2 += pitch2[j];
      offset3 += pitch3[j];
      offset4 += pitch4[j];
      coord[j]++;
      return false;
    } else if (j != 0) {
      offset1 -= (index[j] - 1) * pitch1[j];
      offset2 -= (index[j] - 1) * pitch2[j];
      offset3 -= (index[j] - 1) * pitch3[j];
      offset4 -= (index[j] - 1) * pitch4[j];
      coord[j] = 0;
    } else
      return true;
  }

  //FIXME: use assertion throw "getOffsets Malfunction";
  // To avoid warnings. This point will never be reached.
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

///
/// \brief calculates the lanes for given number of elements
///
///
/// Given a number of elements for a dimension and the data srcType 
/// calculates the number of lanes needed to processing all the elements.
/// A lane has 32 bits (4Bytes) depending on the srcType in use we have
/// have the following table.
///
///   
///      1B  uint8    4 for each lane
///      2B  uint16   2 for each lane
///      4B  uint32   1 for each lane
///      8B  uint64   1/2 for each lane. (We need 2 lanes for represent this 
///                                       srcType)
///
///
/// \tparam[in] srcType, source type of data.
/// \param[out] lanes, Number of lanes needed.
/// \param[out] res, Part of lane to complete all elements.
/// \param[in]  numofelement, All elements for a given dimension to be processed.
///
template <typename srcType>
inline __attribute__((always_inline))
std::pair<int,int>  getLanesResFromNElements(unsigned int numofelements) 
{
  int lanes = 0, res = 0;

  if (getsize<srcType>() == 1) {
    lanes = numofelements / 4;
    res = numofelements - 4 *lanes;
  }
  else if (getsize<srcType>() == 2) {
    lanes = numofelements / 2;
    res = numofelements -2 *lanes;
   }
  else if (getsize<srcType>() == 4) {
      lanes = numofelements;
      res = 0;
  }
  else if (getsize<srcType>() == 8) {
      lanes = numofelements * 2;
      res = 0;
  }

  return std::make_pair(lanes, res);
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
