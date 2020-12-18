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

#ifndef _TOPK_INST_H_
#define _TOPK_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>
#include <climits>

#include "Float16.h"
#include "Writer.h" // From include/internal path
#include "Addresser.h" // From include/internal path
#include "Converter.h" // From include/internal path
#include "Operator.h" // From include/internal path
#include "utils.h" // From include/internal path

namespace dnn_lib {
namespace inlining {
  
inline void swap(void *vals, void *inds, int i, int j) {
  float *fVals = (float *)vals;
  long long *lInds = (long long *)inds;

  float tval = fVals[i];
  long long tind = lInds[i];
  fVals[i] = fVals[j];
  lInds[i] = lInds[j];
  fVals[j] = tval;
  lInds[j] = tind;
}

inline int partition(void *vals, void *inds, int low, int high) {
  float *fVals = (float *)vals;
  long long *lInds = (long long *)inds;

  float pivotVal = fVals[high];
  long long pivotInd = lInds[high];
  int i = low - 1;

  for (int j = low; j <= high - 1; j++) {
    if (fVals[j] != pivotVal) {
      if (fVals[j] > pivotVal) {
        i++;
        dnn_lib::inlining::swap(vals, inds, i, j);
      }
    } else if (lInds[j] < pivotInd) {
      i++;
      dnn_lib::inlining::swap(vals, inds, i, j);
    }
  }
  dnn_lib::inlining::swap(vals, inds, i + 1, high);
  return (i + 1);
}
  
static void partialQuicksort(void *vals, void *inds, int low, int high, int m) {
  if (low < high) {
    int pidx = dnn_lib::inlining::partition(vals, inds, low, high);
    partialQuicksort(vals, inds, low, pidx - 1, m);
    if (pidx < m) {
      partialQuicksort(vals, inds, pidx + 1, high, m);
    }
  }
}

/// \brief Whether the values stored in a and b match.

inline bool same(size_t count, unsigned int a[],  unsigned int b[]) {
  for (size_t index = 0; index < count; ++index) {
    if (a[index] != b[index]) {
      return false;
    }
  }
  return true;
}  
  
template <ElemKind elK>
inline void fwdLibTopKInstThreaded_all(LibTensor* outT, LibTensor* out2T,
                                       LibTensor* inT, unsigned int k,
                                       uint64_t flags,
                                       const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  
  
  /* maintain compatibility through the new Iface Libtensor */
  void* srcT = inT->getRawDataPointer<void>();
  void* dstT = outT->getRawDataPointer<void>();

  const Addresser<elK> inputT(srcT, inT->getScale(), inT->getOffset());
  Addresser<elK> valuesT(dstT, outT->getScale(), outT->getOffset());
  
  long long *indT = out2T->getRawDataPointer<long long>();

  const dim_t *inputIndex = inT->dims().data();
  const dim_t *inputPitch = inT->strides().data();
  const dim_t *valuesIndex = outT->dims().data();
  const dim_t *valuesPitch = outT->strides().data();
  const dim_t *indexPitch = out2T->strides().data();

  int srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  int numElemsValues = valuesPitch[0] * valuesIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsValues, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  for (int i = 0; i < srcDimNum; i++) {
    coord[i] = 0;
  }
  unsigned int l = 0;
  
  /* overloading while sw-2400 and sw-2429 are WIP */
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, valuesPitch, valuesIndex, l);

  int valuesOffset = 0;
  int indexOffset = 0;
  for (int i = 0; i < (int) l; i++) {
    valuesOffset += coord[i] * valuesPitch[i];
    indexOffset += coord[i] * indexPitch[i];
  }
  if (valuesOffset >= numElemsValues)
    return;

  size_t n = inputIndex[srcDimNum - 1];
  unsigned int scratchCoords[3] = {UINT_MAX, UINT_MAX, UINT_MAX};
  // ET_ASSERT(srcDimNum - 1 <= (int)(sizeof(scratchCoords) / sizeof(scratchCoords[0]));
  float tmpValues[n];
  long long tmpInd[n];

  int posMax = initialAddr + maxRead;
  bool done = false;

  while (!done && valuesOffset < posMax) {

    // Recompute the scratch only when a coordinate other than the last changed
    if (not same(srcDimNum - 1, coord, scratchCoords)) {
      int offsetInput = 0;
      // Point to the first element in the innermost dimension
      for (int i = 0; i < srcDimNum - 1; i++) {
        offsetInput += coord[i] * inputPitch[i];
      }
      // Fill the scratch with values and indidces
      for (int i = 0; i < (int) n; i++) {
        tmpValues[i] = inputT[offsetInput + i * inputPitch[srcDimNum - 1]];
        tmpInd[i] = i;
      }
      // Apply partial quicksort
      partialQuicksort(tmpValues, tmpInd, 0, n - 1, k);
      // Save all the coordinates but last
      for (int i = 0; i < srcDimNum - 1; i++) {
        scratchCoords[i] = coord[i];
      }
    }

    int resCoord = coord[srcDimNum - 1];
    valuesT[valuesOffset] = tmpValues[resCoord];
    indT[indexOffset] = tmpInd[resCoord];

    /* overloading while sw-2400 and sw-2429 are WIP */   
    done = getOffsets(srcDimNum, coord, valuesOffset, indexOffset, valuesIndex,
                      valuesPitch, indexPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = (maxRead * typeSize + CACHE_LINE_BYTES - 1) / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

// ONLY FOR K = 1, 2, 3, 4
template <ElemKind elK>
inline void fwdLibTopKInstThreaded_k4(LibTensor* outT, LibTensor* out2T,
                                      LibTensor* inT, unsigned int k,
                                      uint64_t flags,
                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  //  using srcType = typename elemKind2elemTy<elK>::type;
  
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

  /* maintain compatibility through the new Iface Libtensor */
  void* srcT = inT->getRawDataPointer<void>();
  void* dstT = outT->getRawDataPointer<void>();

  const Addresser<elK> inputT(srcT, inT->getScale(), inT->getOffset());
  Addresser<elK> valuesT(dstT, outT->getScale(), outT->getOffset());

  //  long long *indT = (long long *)dstT2;
  long long *indT = out2T->getRawDataPointer<long long>();
  // srcType *valT = (srcType *)dstT;

  // unsigned int *inputIndex = (unsigned int *)srcDims;
  const dim_t *inputIndex = inT->dims().data();

  // unsigned int *valuesPitch = (unsigned int *)dstPitches;
  const dim_t *valuesPitch = outT->strides().data();
  // unsigned int *indexPitch = (unsigned int *)dst2Pitches;
  const dim_t *indexPitch = out2T->strides().data();
  // unsigned int *inputPitch = (unsigned int *)srcPitches;
  const dim_t *inputPitch = inT->strides().data();
  
  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
   
  unsigned int row_length = inputIndex[srcDimNum - 1];
  unsigned int rows = 1;
  for (size_t i = 0; i < size_t(srcDimNum - 1); i++)
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
  indT += batch_offset * indexPitch[batchDim];
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

  int32_t gather_values[] = {0, 4, 8, 12, 0, 4, 8, 12};
  int32_t gather_indices[] = {0, 8, 16, 24, 4, 12, 20, 28};
  __asm__ __volatile__("flw.ps  f31, %[gather_values]\n"
                       "fgw.ps f0, f31(%[tmpValues])\n"
                       "flw.ps  f31, %[gather_indices]\n"
                       "fgw.ps f1, f31(%[tmpInd])\n"

                       :
                       : [ gather_values ]  "m"( *(const int32_t(*)[8]) gather_values),
                         [ gather_indices ] "m"( *(const int32_t(*)[8]) gather_indices),
                         [ tmpValues ] "r"(tmpValues), [ tmpInd ] "r"(tmpInd)
                       : "f31", "f0", "memory");

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
    int32_t gather_coord[] = {0, 8, 16, 24, 4, 12, 20, 28};
    float tmpT[4];
    __asm__ __volatile__("flw.ps  f31, %[gather_coord]\n"
                         "fscw.ps f1, f31(%[indT])\n"
                         "mov.m.x m0, zero, 0x0f\n"
                         "fsw.ps f0, %[tmpT]\n"
                         : [ tmpT ] "=m"( *(float(*)[4]) tmpT)
                         : [ indT ] "r"(indT),
                           [ gather_coord ] "m"( *(const int32_t(*)[8]) gather_coord)
                         : "f0", "f1", "f31", "memory");
    for (unsigned i = 0; i < k; i++)
      valuesT[batch_offset * valuesPitch[batchDim] + i] = tmpT[i];
  }
}

// ONLY FOR K = 5, 6, 7, 8
template <ElemKind elK>
inline void fwdLibTopKInstThreaded_k8(LibTensor* outT, LibTensor* out2T, LibTensor* inT,
                                      unsigned int k, uint64_t flags,
                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  //  using srcType = typename elemKind2elemTy<elK>::type;
  
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* srcT = inT->getRawDataPointer<void>();
  void* dstT = outT->getRawDataPointer<void>();
  
  const Addresser<elK> inputT(srcT, inT->getScale(), inT->getOffset());
  Addresser<elK> valuesT(dstT, outT->getScale(), outT->getOffset());
  
  //long long *indT = (long long *)dstT2;
  long long *indT = out2T->getRawDataPointer<long long>();
  
  // unsigned int *inputIndex = (unsigned int *)srcDims;
  const dim_t *inputIndex = inT->dims().data();
  // unsigned int *valuesPitch = (unsigned int *)dstPitches;
  const dim_t *valuesPitch = outT->strides().data();
  // unsigned int *indexPitch = (unsigned int *)dst2Pitches;
  const dim_t *indexPitch = out2T->strides().data();
  // unsigned int *inputPitch = (unsigned int *)srcPitches;
  const dim_t *inputPitch = inT->strides().data();

  uint8_t srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  unsigned int row_length = inputIndex[srcDimNum - 1];
  unsigned int rows = 1;
  for (uint8_t i = 0; i < srcDimNum - 1; i++)
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
  indT += batch_offset * indexPitch[batchDim];

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
  __asm__ __volatile__("flw.ps  f31, %[gather_values]\n"
                       "fgw.ps f0, f31(%[tmpValues])\n"
                       "faddi.pi f31, f31, 0x10\n" // New gather = {16, 20, 24,
                                                   // 28, 16, 20, 24, 28}
                       "fgw.ps f1, f31(%[tmpValues])\n"
                       "flw.ps  f31, %[gather_indices]\n"
                       "fgw.ps f2, f31(%[tmpInd])\n"
                       "faddi.pi f31, f31, 0x20\n" // New gather = {32, 40, 48,
                                                   // 56, 36, 44, 52, 60}
                       "fgw.ps f3, f31(%[tmpInd])\n"

                       :
                       : [ gather_values ]  "m"( *(const int32_t(*)[8]) gather_values),
                         [ gather_indices ] "m"( *(const int32_t(*)[8]) gather_indices),
                         [ tmpValues ] "r"(tmpValues),
                         [tmpValuesMem] "m" (*(const float(*)[9]) tmpValues),
                         [ tmpInd ] "r"(tmpInd),
                         [tmpIndMem] "m" (*(const long long(*)[9]) tmpInd)
                       : "f31", "f0", "f1", "f2", "f3");

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
    int32_t gather_indices[] = {0, 8, 16, 24, 4, 12, 20, 28};
    float tmpT[8];
    __asm__ __volatile__("flw.ps  f31, %[gather_indices]\n"
                         "fscw.ps f2, f31(%[indT])\n"
                         "faddi.pi f31, f31, 0x20\n"
                         "fscw.ps f3, f31(%[indT])\n"
                         "mov.m.x m0, zero, 0x0F\n"
                         "fsw.ps f0, %[tmpT]\n"
                         "fsw.ps f1, 0x10+%[tmpT]\n"

                         : [ tmpT ] "=m"( *(float(*)[8]) tmpT)
                         : [ indT ] "r"(indT),
                           [ gather_indices ] "m"( *(const int32_t(*)[8]) gather_indices),
                           [ gather_values ] "r"(gather_values)
                         : "f0", "f1", "f2", "f3", "f31");
    for (unsigned i = 0; i < k; i++)
      valuesT[batch_offset * valuesPitch[batchDim] + i] = tmpT[i];
  }
}


template <ElemKind elK>
inline void fwdLibTopKInst(LibTensor* outT, LibTensor* out2T,
                           LibTensor* inT, const uint32_t k,
                           uint64_t flags,
                           const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  // TODO : For (k < 4) the specialized version is not working, needs to be fixed. 
  if ((k > 8) or (k < 4))
    fwdLibTopKInstThreaded_all<elK>(outT, out2T, inT, k, flags, minionOffset, assignedMinions);
  else if (k > 4)
    fwdLibTopKInstThreaded_k8<elK>(outT, out2T, inT, k, flags, minionOffset, assignedMinions);
  else
    fwdLibTopKInstThreaded_k4<elK>(outT, out2T, inT, k, flags, minionOffset, assignedMinions);
}

  
} // namespace inlining

} // namespace dnn_lib

#endif // _TOPK_INST_H_ 
