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

#include "Float16.h"
#include "Writer.h" // From include/internal path
#include "Addresser.h" // From include/internal path
#include "Converter.h" // From include/internal path
#include "Operator.h" // From include/internal path
#include "utils.h" // From include/internal path

namespace dnn_lib {

void partialQuicksort(void *vals, void *inds, int low, int high, int m);

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

// In this implementation we suppose that the dstPitches (1 and 2) have padding
// which ensures the dstPitches[n-2] being multiple of cacheline length if not,
// it needs sore global or reduce
template <typename srcType>
inline void fwdLibTopKInst(void *dstT, void *dstDims, void *dstPitches,
                             void *dstT2, void *dst2Dims, void *dst2Pitches,
                             void *srcT, void *srcDims, void *srcPitches,
                             unsigned int srcDimNum, unsigned int k,
                             const float *scale, const int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  const Addresser<srcType> inputT(srcT, scale[0], offset[0]);
  Addresser<srcType> valuesT(dstT, scale[3], offset[3]);

  long long *indT = (long long *)dstT2;

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
inline void fwdLibTopKInstThreaded_all(
    void *dstT, void *dstDims, void *dstPitches, void *dstT2, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, unsigned int k, const float *scale, const int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> inputT(srcT, scale[0], offset[0]);
  Addresser<srcType> valuesT(dstT, scale[3], offset[3]);

  long long *indT = (long long *)dstT2;

  unsigned int *valuesIndex = (unsigned int *)dstDims;
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
  for (unsigned i = 0; i < srcDimNum; i++)
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
void fwdLibTopKInstThreaded_k4(void *dstT, void *dstDims, void
*dstPitches, void *dstT2, void *dst2Dims, void *dst2Pitches, void *srcT, void
*srcDims, void *srcPitches, unsigned int srcDimNum, unsigned int k, const float
*scale, const int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  __asm__ __volatile__ ("mov.m.x m0, zero, 0xff \n");

  const Addresser<srcType> inputT(srcT, scale[0], offset[0]);
  Addresser<srcType> valuesT(dstT, scale[3], offset[3]);

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
inline void fwdLibTopKInstThreaded_k4(
    void *dstT, void *dstDims, void *dstPitches, void *dstT2, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, unsigned int k, const float *scale, const int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

  const Addresser<srcType> inputT(srcT, scale[0], offset[0]);
  Addresser<srcType> valuesT(dstT, scale[3], offset[3]);

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
template <typename srcType>
inline void fwdLibTopKInstThreaded_k8(
    void *dstT, void *dstDims, void *dstPitches, void *dstT2, void *dst2Dims,
    void *dst2Pitches, void *srcT, void *srcDims, void *srcPitches,
    unsigned int srcDimNum, unsigned int k, const float *scale, const int32_t *offset,
    uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> inputT(srcT, scale[0], offset[0]);
  Addresser<srcType> valuesT(dstT, scale[3], offset[3]);

  long long *indT = (long long *)dstT2;

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

} // namespace inlining

} // namespace dnn_lib

#endif // _TOPK_INST_H_ 
