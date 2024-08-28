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

#ifndef UTILS_H
#define UTILS_H

#include "Float16.h"
#include "LibCommon.h"
#include <cmath>
#include <cstdint>
#include <etsoc/common/utils.h>
#include <etsoc/isa/cacheops-umode.h>
#include <etsoc/isa/hart.h>
#include <etsoc/isa/tensors.h>
#include <sstream>
#include <utility>

namespace dnn_lib {

constexpr float kLog2E = static_cast<float>(1.44269504088896340736);
constexpr float kRecLog2E = static_cast<float>(1.0 / 1.44269504088896340736);

#define SET_MINUS_INFTY(_reg) "fbci.ps " #_reg ", 0xff800 \n" // _reg is vect

#define print(s) et_printf((s))

//-------------------------------------------------------------------------------------------------
//
// FUNCTION: fence_evict_va
//
// just calls evict_va, but doing a fence first
inline __attribute__((always_inline)) void fence_evict_va(uint64_t use_tmask, uint64_t dst, uint64_t addr,
                                                          uint64_t num_lines = 0, uint64_t stride = 0,
                                                          uint64_t id = 0) {
  FENCE;
  cache_ops_evict_va(use_tmask, dst, addr, num_lines, stride, id);
}

//-------------------------------------------------------------------------------------------------
//
// FUNCTION: evict_va_multi
//
//   This function is a wrapper of evict_va for any number for cache lines. It calls evict_va as
//   many times as needed to evict all lines
//
inline __attribute__((always_inline)) void evict_va_multi(uint64_t dst, uintptr_t addr, size_t num_lines) {
  FENCE;
  while (num_lines > 16) {
    cache_ops_evict_va(0, dst, addr, 15, CACHE_LINE_BYTES, 0);
    addr += (CACHE_LINE_BYTES * 16);
    num_lines -= 16;
  }
  if (num_lines > 0)
    cache_ops_evict_va(0, dst, addr, static_cast<uint64_t>(num_lines - 1), CACHE_LINE_BYTES, 0);
}

inline __attribute__((always_inline)) unsigned int gcd(unsigned int a, unsigned int b) {
  if (b == 1)
    return a;
  return gcd(b, a % b);
}

template <class T> constexpr std::size_t getsize() {
  return sizeof(T);
}
template <> constexpr std::size_t getsize<float16>() {
  return 2;
}

template <> constexpr std::size_t getsize<bfloat16>() {
  return 2;
}

/**
 * @brief Converts an offset in a tensor into its corresponding coordinates.
 *
 * This function takes into account the padding that the tensor might have, but if
 * the offset corresponds to a padding position, the returned coordinates will
 * point this padding position in the matrix (outside the dimensions).
 *
 * @warning The function cannot be used with broadcast tensors (which have some
 * pitch equal to 0).
 *
 * @param[out] coord Vector that will be filled with the coordinates of the
 *  offset in the tensor.
 * @param[in] offset Unsigned integer referring to a position in a tensor.
 * @param[in] srcDimNum The "number of dimensions" of the tensor.
 * @param[in] pitch The vector of pitches of the given tensor.
 */

/* overloading while sw-2400 and sw-2429 are WIP */
template <typename T>
inline __attribute__((always_inline)) void getCoordinates(unsigned int* coord, unsigned int offset, unsigned int dimNum,
                                                          const T* pitch) {
  unsigned int rm = offset;
  for (unsigned int i = 0; i < dimNum; i++) {
    assert((pitch[i] != 0) and "Broadcast pitch 0 not supported");
    coord[i] = rm / pitch[i];
    rm = rm - coord[i] * pitch[i];
  }
}

/* New function signature, to replace all others at the end of SW-11349 and SW-11753 */
template <typename T>
inline __attribute__((always_inline)) void getCoordinates(dim_array_t& coord, size_t offset, dim_t dimNum,
                                                          const T* pitch) {
  auto rm = offset;
  for (dim_t i = 0; i < dimNum; i++) {
    assert((pitch[i] != 0) and "Broadcast pitch 0 not supported");
    coord[i] = rm / pitch[i];
    rm = rm - coord[i] * pitch[i];
  }
}

/**
 * @brief Advances offset and coordinates by one element.
 *
 * This function takes into account the padding that the tensor might have.
 *
 * @param[inout] coord Vector that will be updated to the next valid element in the tensor.
 * @param[inout] offset referring to the position in the tensor accounting padding.
 * @param[in] dims The "number of dimensions" of the tensor.
 * @param[in] pitch The vector of pitches of the given tensor.
 */

inline __attribute__((always_inline)) void advanceOffsetAndCoordinates(dim_array_t& coord, const dim_array_t& dims,
                                                                       size_t& offset, dim_t dimNum,
                                                                       const dim_array_t& pitch) {
  for (int j = dimNum - 1; j >= 0; j--) {
    if (coord[j] != (dims[j] - 1)) {
      offset += pitch[j];
      coord[j]++;
      break;
    } else if (j != 0) {
      // previous iteration was in last element of dimension j, reset that dimension
      offset -= (dims[j] - 1) * pitch[j];
      coord[j] = 0;
    }
  }
}

/**
 * @brief Converts an offset in a tensor into its corresponding non-padding-coords.
 *
 * This function takes into account the padding that the tensor might have, and if
 * the offset corresponds to a padding position, the returned coordinates will
 * point the next position in the tensor that doesn't correspond to padding.
 *
 * @warning The function cannot be used with broadcast tensors (which have some
 * pitch equal to 0).
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

template <typename pitch_t, typename dims_t>
inline __attribute__((always_inline)) void getNonPaddingCoordinates(unsigned int* coord, unsigned int offset,
                                                                    unsigned int srcDimNum, const pitch_t* pitch,
                                                                    const dims_t* dims, unsigned int& k) {

  getCoordinates(coord, offset, srcDimNum, pitch);
  k = srcDimNum;
  for (int j = srcDimNum - 1; j > 0; j--) {
    if (unlikely(coord[j] >= dims[j])) {
      coord[j - 1]++;
      k = j;
    }
  }
  for (unsigned int j = k; j < srcDimNum; j++) {
    coord[j] = 0;
  }
}

/* New function signature, to replace the function above at the end of SW-11349 and SW-11753 */
template <typename pitch_t, typename dims_t>
inline __attribute__((always_inline)) void getNonPaddingCoordinates(dim_array_t& coord, size_t offset, dim_t srcDimNum,
                                                                    const pitch_t* pitch, const dims_t* dims,
                                                                    dim_t& k) {

  getCoordinates(coord, offset, srcDimNum, pitch);
  k = srcDimNum;
  for (sdim_t j = srcDimNum - 1; j > 0; j--) {
    if (unlikely(coord[j] >= dims[j])) {
      coord[j - 1]++;
      k = j;
    }
  }
  for (dim_t j = k; j < srcDimNum; j++) {
    coord[j] = 0;
  }
  offset = 0;
  for (dim_t d = 1; d < srcDimNum; d++) {
    offset += coord[d] * pitch[d];
  }
}

/**
 * @brief Given a tensor, it divides it in elements for the minions.
 *
 * It gives to each minion an offset to start and how many elements to work on.
 * The division ensures that all active minions minions (except for, possibly,
 * the last one) work with the same number of elements and the following
 * have no positions to work with.
 *
 * @warning The number maxRead does not take into account padding, so if maxRead
 *  is 16, that does not mean that the minion has to work on 16 elements: some of
 *  them may be padding. Moreover, @f$ offset + maxRead@f$ may be outside the tensor.
 *
 * @warning The function works with the supposition that the minions working on this
 *  tensor is numbered from 0 to activeMinions.
 *
 * @param[in] numElems The number of elements in the tensor that is divided.
 * @param[out] offset The starting offset for the minion.
 * @param[out] maxRead The number of consecutive elements the minion is assigned.
 * @param[in] minionId The id of the minion that calls the function.
 * @param[in] activeMinions The number of minions that is working on the tensor.
 */
inline __attribute__((always_inline)) void getGlobalPartition(unsigned int numElems, unsigned int& offset,
                                                              unsigned int& maxRead, unsigned int minionId,
                                                              unsigned int activeMinions) {

  // Ensure that all the minions have a least one element to do
  if (unlikely(activeMinions > numElems)) {
    activeMinions = numElems;

    // When there is an excess of minions make redundant the ones for which there is no work
    if (unlikely(minionId >= activeMinions)) {
      offset = 0;
      maxRead = 0;
      return;
    }
  }

  // Each minion will process a number of consecutive elements (a region)
  unsigned int regionSize = numElems / activeMinions;

  // After covering with "activeMinions" regions, each region containing "regionSize"
  // numElems, there is still a remainder.
  unsigned int elemsRemainder = numElems % activeMinions;

  // The remainder of elements is done by adding one element to each minion whose
  // id is greater or equal than firstMinionDoingOneExtra. For example, if the
  // remainder is 3 elements and the number of active minions is 4, then minions 1,
  // 2 and 3 should do an extra element.
  unsigned int firstMinionDoingOneExtra = activeMinions - elemsRemainder;

  if (minionId < firstMinionDoingOneExtra) {
    maxRead = regionSize;
    offset = regionSize * minionId;
  } else {
    maxRead = regionSize + 1;
    offset = regionSize * firstMinionDoingOneExtra + maxRead * (minionId - firstMinionDoingOneExtra);
  }

  if (unlikely(offset >= numElems)) {
    // Do nothing when offset is beyond numElems minus one
    maxRead = 0;
  } else {
    // Clip maxRead so that offset plus maxRead does not got beyond numElems minus one
    maxRead = std::min(maxRead, numElems - offset);
  }
}

/* New function signature, to replace the function above at the end of SW-11349 and SW-11753 */
inline __attribute__((always_inline)) void getGlobalPartition(size_t numElems, size_t& offset, size_t& maxRead,
                                                              size_t minionId, size_t activeMinions) {

  // Ensure that all the minions have a least one element to do
  if (unlikely(activeMinions > numElems)) {
    activeMinions = static_cast<uint32_t>(numElems);

    // When there is an excess of minions make redundant the ones for which there is no work
    if (unlikely(minionId >= activeMinions)) {
      offset = 0;
      maxRead = 0;
      return;
    }
  }

  // Each minion will process a number of consecutive elements (a region)
  auto regionSize = numElems / activeMinions;

  // After covering with "activeMinions" regions, each region containing "regionSize"
  // numElems, there is still a remainder.
  auto elemsRemainder = numElems % activeMinions;

  // The remainder of elements is done by adding one element to each minion whose
  // id is greater or equal than firstMinionDoingOneExtra. For example, if the
  // remainder is 3 elements and the number of active minions is 4, then minions 1,
  // 2 and 3 should do an extra element.
  auto firstMinionDoingOneExtra = activeMinions - elemsRemainder;

  if (minionId < firstMinionDoingOneExtra) {
    maxRead = regionSize;
    offset = regionSize * minionId;
  } else {
    maxRead = regionSize + 1;
    offset = regionSize * firstMinionDoingOneExtra + maxRead * (minionId - firstMinionDoingOneExtra);
  }

  if (unlikely(offset >= numElems)) {
    // Do nothing when offset is beyond numElems minus one
    maxRead = 0;
  } else {
    // Clip maxRead so that offset plus maxRead does not got beyond numElems minus one
    maxRead = std::min(maxRead, numElems - offset);
  }
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
 * @param[in] elementSize The number of bytes of each element in the matrix.
 *  It is required to be a power of 2 and smaller than 64 (1, 2, 4, 8, usually).
 * @param[in] numElems The number of elements in the tensor that is divided.
 * @param[out] offset The starting offset for the minion.
 * @param[out] maxRead The number of consecutive elements the minion is assigned.
 * @param[in] minionId The id of the minion that calls the function.
 * @param[in] activeMinions The number of minions that is working on the tensor.
 */
inline __attribute__((always_inline)) void getCachelinePartition(size_t elementSize, size_t numElems, size_t& offset,
                                                                 size_t& maxRead, size_t minionId, size_t activeMinions,
                                                                 void* addr) {

  // How many elements does a cache line contain
  auto cacheLineSizeElems = CACHE_LINE_BYTES / elementSize;

  // When unaligned, how many elements from the tensor are in the first cache line,
  // or zero when aligned.
  auto unalignedElements =
    ((((uintptr_t)addr + CACHE_LINE_BYTES - 1) & ~(CACHE_LINE_BYTES - 1)) - (uintptr_t)addr) / elementSize;

  // When unaligned, how many elements from the first cache line do not belong to the tensor,
  // or zero when aligned.
  auto missalignmentElements = likely(unalignedElements == 0) ? 0 : cacheLineSizeElems - unalignedElements;

  // Total number of cache lines (rounded up)
  auto totalCacheLines = (missalignmentElements + numElems - 1) / cacheLineSizeElems + 1;

  // Ensure that all the minions have a least one cache line to do
  if (unlikely(activeMinions > totalCacheLines)) {
    activeMinions = totalCacheLines;

    // When there is an excess of minions make redundant the ones for which there is no work
    if (unlikely(minionId >= activeMinions)) {
      offset = 0;
      maxRead = 0;
      return;
    }
  }

  // Each minion will process a number of consecutive cache lines (a region)
  auto regionSizeLines = totalCacheLines / activeMinions;

  // After covering with "activeMinions" regions, each region containing "regionSizeLines"
  // lines, there is still a cache lines remainder.
  auto cacheLinesRemainder = totalCacheLines % activeMinions;

  // The remainder of cache lines is done by adding one extra line to each minion whose
  // id is greater or equal than firstMinionDoingOneExtra. For example, if the
  // remainder is 3 lines and the number of active minions is 4, then minions 1,
  // 2 and 3 should do an extra cache line.
  auto firstMinionDoingOneExtra = activeMinions - cacheLinesRemainder;

  if (minionId < firstMinionDoingOneExtra) {
    maxRead = regionSizeLines;
    offset = regionSizeLines * minionId;
  } else {
    maxRead = regionSizeLines + 1;
    offset = regionSizeLines * firstMinionDoingOneExtra + maxRead * (minionId - firstMinionDoingOneExtra);
  }

  // Convert from cache lines to elements
  maxRead *= cacheLineSizeElems;
  offset *= cacheLineSizeElems;

  // Ensure minions other than zero start on a cache line boundary
  if (unlikely(missalignmentElements > 0 and activeMinions > 1)) {
    if (minionId == 0) {
      // Minion zero does "missAlignmentElements" fewer elements
      maxRead -= missalignmentElements;
    } else if (likely(minionId != activeMinions - 1)) {
      // Minions that are neither zero or last move "missAlignmentElements" left
      offset -= missalignmentElements;
    } else {
      // The last minion moves "missAlignmentElements" left and does "missAlignmentElements" extra elements
      offset -= missalignmentElements;
      maxRead += missalignmentElements;
    }
  }

  if (unlikely(offset >= numElems)) {
    // Do nothing when offset is beyond numElems minus one
    maxRead = 0;
  } else {
    // Clip maxRead so that offset plus maxRead does not got beyond numElems minus one
    maxRead = std::min(maxRead, numElems - offset);
  }
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
inline __attribute__((always_inline)) void getReversedCachelinePartition(size_t elementsize, size_t ElemsDst,
                                                                         size_t& offset, size_t& maxRead,
                                                                         size_t activeMinions) {
  // TODO : Needs to take into account unaligned destination tensor.
  //        Needs to use minionId from operator with substracted initial minion.
  //        Not sure why this version is require, only used in LengthsToRanges operator.
  //        This version seems to set the extra cacheline in the first minion,
  //        rather than the last minion.
  size_t minionId = (activeMinions - get_minion_id()) - 1;
  auto cll = CACHE_LINE_BYTES / elementsize; // Cacheline length
  auto ncl = (ElemsDst - 1) / cll + 1;       // Amount of cache lines
  auto mcl = ncl / activeMinions;            // Amount of cl for a minion
  auto mod = ncl - activeMinions * mcl;
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
template <typename offset_t, typename dims_t, typename pitches_t>
inline __attribute__((always_inline)) bool getOffsets(dim_t dimNum, dim_array_t& coord, offset_t& offset,
                                                      const dims_t* index, const pitches_t* pitch) {

  for (sdim_t j = dimNum - 1; j >= 0; j--) {
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

  // FIXME: use assertion throw "getOffsets Malfunction";
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
inline __attribute__((always_inline)) bool getOffsets(dim_t dimNum, dim_array_t& coord, T& offset1, T& offset2,
                                                      const dim_t* index, const dim_t* pitch1, const dim_t* pitch2) {

  for (sdim_t j = dimNum - 1; j >= 0; j--) {
    if (likely(coord[j] != (index[j] - 1))) {
      offset1 += pitch1[j];
      offset2 += pitch2[j];
      coord[j]++;
      return false;
    } else if (likely(j != 0)) {
      offset1 -= (index[j] - 1) * pitch1[j];
      offset2 -= (index[j] - 1) * pitch2[j];
      coord[j] = 0;
    } else {
      return true;
    }
  }

  // FIXME: use assertion throw "getOffsets Malfunction";
  // To avoid warnings. This point will never be reached.
  return true;
}

/* overloading while sw-2400 and sw-2429 are WIP */
template <typename T, typename U, typename S>
inline __attribute__((always_inline)) bool getOffsets(dim_t dimNum, dim_array_t& coord, T& offset1, T& offset2,
                                                      U* index, S* pitch1, U* pitch2) {

  for (sdim_t j = dimNum - 1; j >= 0; j--) {
    if (likely(coord[j] != (index[j] - 1))) {
      offset1 += static_cast<T>(pitch1[j]);
      offset2 += static_cast<T>(pitch2[j]);
      coord[j]++;
      return false;
    } else if (likely(j != 0)) {
      offset1 -= static_cast<T>((index[j] - 1) * pitch1[j]);
      offset2 -= static_cast<T>((index[j] - 1) * pitch2[j]);
      coord[j] = 0;
    } else
      return true;
  }

  // FIXME: use assertion throw "getOffsets Malfunction";
  // To avoid warnings. This point will never be reached.
  return true;
}

/* overloading while sw-2400 and sw-2429 are WIP */
inline __attribute__((always_inline)) bool getOffsets(dim_t dimNum, dim_array_t& coord, size_t& offset1,
                                                      size_t& offset2, const dim_t* index, const dim_t* pitch1,
                                                      dim_t* pitch2) {

  for (sdim_t j = dimNum - 1; j >= 0; j--) {
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

  // FIXME: use assertion throw "getOffsets Malfunction";
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
template <typename T, typename U, typename S>
inline __attribute__((always_inline)) bool getOffsets(dim_t dimNum, dim_array_t& coord, T& offset1, T& offset2,
                                                      T& offset3, U* index, U* pitch1, U* pitch2, S* pitch3) {

  for (sdim_t j = dimNum - 1; j >= 0; j--) {
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

  // FIXME: use assertion throw "getOffsets Malfunction";
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
template <typename offset_t, typename dims_t, typename pitches_t>
inline __attribute__((always_inline)) bool
getOffsets(dim_t dimNum, dim_array_t& coord, offset_t& offset1, offset_t& offset2, offset_t& offset3, offset_t& offset4,
           const dims_t* index, const pitches_t* pitch1, const pitches_t* pitch2, const pitches_t* pitch3,
           const pitches_t* pitch4) {

  for (sdim_t j = dimNum - 1; j >= 0; j--) {
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

  // FIXME: use assertion throw "getOffsets Malfunction";
  // To avoid warnings. This point will never be reached.
  return true;
}

constexpr double kE = 2.71828182845904523536;

template <bool setMask = true> inline __attribute__((always_inline)) float getExp(float val) {
  float ret;
  if (val == 0) {
    return 1;
  } else if (val > 0) {
    dnn_lib::fpPowSingleElement<setMask>(static_cast<float>(kE), val, ret);
    return ret;
  } else { // val<0
    dnn_lib::fpPowSingleElement<setMask>(static_cast<float>(kE), -val, ret);
    dnn_lib::fpReciprocalSingleElement<setMask>(ret, ret);
    return ret;
  }
}

template <bool setMask = true> inline __attribute__((always_inline)) float getSinh(float val) {
  float op1, op2;
  if (val == 0) {
    op1 = op2 = 1;
  } else if (val > 0) {
    dnn_lib::fpPowSingleElement<setMask>(kE, val, op1);
    dnn_lib::fpReciprocalSingleElement<setMask>(op1, op2);
  } else { // val<0
    dnn_lib::fpPowSingleElement<setMask>(kE, -val, op2);
    dnn_lib::fpReciprocalSingleElement<setMask>(op2, op1);
  }
  return 0.5f * (op1 - op2);
}

template <bool setMask = true> inline __attribute__((always_inline)) float getCosh(float val) {
  float op1, op2;
  if (val == 0) {
    op1 = op2 = 1;
  } else if (val > 0) {
    dnn_lib::fpPowSingleElement<setMask>(kE, val, op1);
    dnn_lib::fpReciprocalSingleElement<setMask>(op1, op2);
  } else { // val<0
    dnn_lib::fpPowSingleElement<setMask>(kE, val, op2);
    dnn_lib::fpReciprocalSingleElement<setMask>(op2, op1);
  }
  return 0.5f * (op1 + op2);
}

template <bool setMask = true> inline __attribute__((always_inline)) float getTanh(float val) {
  if (val > 10) {
    return 1;
  } else if (val < -10) {
    return -1;
  } else {
    float e2x, denom;
    dnn_lib::fpPowSingleElement<setMask>(static_cast<float>(kE), 2 * val, e2x);
    dnn_lib::fpReciprocalSingleElement<setMask>(e2x + 1, denom);
    return (denom * (e2x - 1));
  }
}

template <bool setMask = true> inline __attribute__((always_inline)) float getSin(float val) {
  return sinf(val);
}

template <bool setMask = true> inline __attribute__((always_inline)) float getCos(float val) {
  return cosf(val);
}

inline __attribute__((always_inline)) bool isInteger(float f) {
  return (nearbyintf(f) == f);
}

inline __attribute__((always_inline)) bool isEven(int val) {
  return ((val % 2) == 0);
}

template <bool setMask = true> inline __attribute__((always_inline)) float getPow(float base, float exp) {
  float dst_tmp, inverted;
  if (base == 0)
    return (exp == 0);
  else if ((base < 0) && !isInteger(exp))
    return NAN;
  else if ((base > 0) && exp == 0)
    return 1;
  else if (exp > 0) {
    if (base < 0) {
      dnn_lib::fpPowSingleElement<setMask>(-base, exp, dst_tmp);
      if (isEven((int)exp))
        return dst_tmp;
      else
        return -dst_tmp;
    } else {
      dnn_lib::fpPowSingleElement<setMask>(base, exp, dst_tmp);
      return dst_tmp;
    }
  } else if (exp < 0) {
    dnn_lib::fpPowSingleElement<setMask>(base, -exp, inverted);
    dnn_lib::fpReciprocalSingleElement<setMask>(inverted, dst_tmp);
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
inline __attribute__((always_inline)) std::pair<int, int> getLanesResFromNElements(uint32_t numofelements) {
  int lanes = 0, res = 0;

  if (getsize<srcType>() == 1) {
    lanes = numofelements / 4;
    res = numofelements - 4 * lanes;
  } else if (getsize<srcType>() == 2) {
    lanes = numofelements / 2;
    res = numofelements - 2 * lanes;
  } else if (getsize<srcType>() == 4) {
    lanes = numofelements;
    res = 0;
  } else if (getsize<srcType>() == 8) {
    lanes = numofelements * 2;
    res = 0;
  }

  return std::make_pair(lanes, res);
}

//@TODO: REPLACE template by const size_t once all operands are using LibTensor
template <typename T>
inline __attribute__((always_inline)) bool getNextStep(unsigned int dimNum, unsigned int* coord, T* dims) {
  for (unsigned int i = 0; i < dimNum; i++) {
    if (coord[i] < dims[i] - 1) {
      coord[i] = coord[i] + 1;
      break;
    } else {
      if (i == dimNum - 1) {
        return true;
      } else {
        coord[i] = 0;
      }
    }
  }
  return false;
}

/* New function signature, to replace the function above at the end of SW-11349 and SW-11753 */
template <typename T>
inline __attribute__((always_inline)) bool getNextStep(dim_t dimNum, dim_array_t& coord, T* dims) {
  for (dim_t i = 0; i < dimNum; i++) {
    if (coord[i] < dims[i] - 1) {
      coord[i] = coord[i] + 1;
      break;
    } else {
      if (i == dimNum - 1) {
        return true;
      } else {
        coord[i] = 0;
      }
    }
  }
  return false;
}

/* Routine getOffset seeems deprecated, cant find any reference */

template <typename T>
inline __attribute__((always_inline)) unsigned int getOffset(unsigned int* coord, unsigned int dimNum, const T* pitch) {
  unsigned int offset = 0;
  for (unsigned int i = 0; i < dimNum; i++) {
    offset += coord[i] * pitch[i];
  }
  return offset;
}

/* New function signature, to replace the function above at the end of SW-11349 and SW-11753 */
template <typename T>
inline __attribute__((always_inline)) size_t getOffset(dim_array_t& coord, dim_t dimNum, const T* pitch) {
  size_t offset = 0;
  for (dim_t i = 0; i < dimNum; i++) {
    offset += coord[i] * pitch[i];
  }
  return offset;
}

} // namespace dnn_lib
#endif /* UTILS_H */
