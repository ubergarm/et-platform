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

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "LibNodes.h"
#include "GenInstances.h"
#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"

using namespace std;

void dnn_lib::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTy(
    void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
    void *pindices, void *plengths, unsigned int pLengthsSize) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  float *tOutput = (float *)pdst;
  uint8_t *tAInput = (uint8_t *)pdata;
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
  uint8_t *tAInput = (uint8_t *)pdata;
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

template<bool Int8Src, bool Float16Dst>
void dnn_lib::
    fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized(
        void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
	    void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
	    void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
	    void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags,
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
  float   *scales  = (float   *) pscale;
  float   *offsets = (float   *) poffset;

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
  uintptr_t dstElemSize;
  if (Float16Dst)  // For dual output use float32 blocking for the tail
    dstElemSize = 2;
  else
    dstElemSize = 4;

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
        : "t0", "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
          "f30", "f31"
      );

      if (Float16Dst) {
        // Set offsets for storing float16 results (0, 2, 4, 6, 8, 10, 12, 14)
        __asm__ __volatile__ (
          "fadd.pi f29, f31, f31\n"
          :
          :
          : "f29"
        );
      }

      // For all sparse input rows.
      for (uintptr_t j = 0, currIndex = minionCurrIndex;
           j < currSegmentLength; j++, currIndex++) {
		int64_t            rowIndex   = indices[currIndex];
        volatile uint8_t * data_ptr   = tAInput + rowIndex * dataPitches[0];
        float            * scale_ptr  = (float *) &scales[rowIndex];
        float            * offset_ptr = (float *) &offsets[rowIndex];
        float            * weight_ptr = (float *) &tWInput[currIndex];

        __asm__ __volatile__ (
          "fbc.ps  f26, 0x0(%[weight_ptr])\n"
          "fbc.ps  f27, 0x0(%[offset_ptr])\n"
          "fbc.ps  f28, 0x0(%[scale_ptr])\n"

          // Load a full input cache line (64 elements, 8 vregs)
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
          "fgb.ps     f18, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
         : [data_ptr]   "+&r" (data_ptr)
         : [weight_ptr] "r"   (weight_ptr),
           [offset_ptr] "r"   (offset_ptr),
           [scale_ptr]  "r"   (scale_ptr)
         : "f18" , "f19", "f20", "f21", "f22", "f23", "f24",
           "f25", "f26", "f27", "f28"
        );

        if (Int8Src) {
          // Convert to UInt8_t adding 128.
        	__asm__ __volatile__ (
            "faddi.pi f25, f25, 0x80\n"
            "faddi.pi f24, f24, 0x80\n"
            "faddi.pi f23, f23, 0x80\n"
            "faddi.pi f22, f22, 0x80\n"
            "faddi.pi f21, f21, 0x80\n"
            "faddi.pi f20, f20, 0x80\n"
            "faddi.pi f19, f19, 0x80\n"
            "faddi.pi f18, f18, 0x80\n"
           :
           : 
           : "f18" , "f19", "f20", "f21", "f22", "f23", "f24",
             "f25"
          );
        }

        __asm__ __volatile__ (
          "fand.pi    f25, f25, f30\n"
          "fand.pi    f24, f24, f30\n"
          "fand.pi    f23, f23, f30\n"
          "fand.pi    f22, f22, f30\n"
          "fand.pi    f21, f21, f30\n"
          "fand.pi    f20, f20, f30\n"
          "fand.pi    f19, f19, f30\n"
          "fand.pi    f18, f18, f30\n"
          "fcvt.ps.pw f25, f25\n"
          "fcvt.ps.pw f24, f24\n"
          "fcvt.ps.pw f23, f23\n"
          "fcvt.ps.pw f22, f22\n"
          "fcvt.ps.pw f21, f21\n"
          "fcvt.ps.pw f20, f20\n"
          "fcvt.ps.pw f19, f19\n"
          "fcvt.ps.pw f18, f18\n"
          "fmadd.ps   f25, f25, f28, f27\n"
          "fmadd.ps   f24, f24, f28, f27\n"
          "fmadd.ps   f23, f23, f28, f27\n"
          "fmadd.ps   f22, f22, f28, f27\n"
          "fmadd.ps   f21, f21, f28, f27\n"
          "fmadd.ps   f20, f20, f28, f27\n"
          "fmadd.ps   f19, f19, f28, f27\n"
          "fmadd.ps   f18, f18, f28, f27\n"
          "fmadd.ps   f0, f26, f25, f0\n"
          "fmadd.ps   f1, f26, f24, f1\n"
          "fmadd.ps   f2, f26, f23, f2\n"
          "fmadd.ps   f3, f26, f22, f3\n"
          "fmadd.ps   f4, f26, f21, f4\n"
          "fmadd.ps   f5, f26, f20, f5\n"
          "fmadd.ps   f6, f26, f19, f6\n"
          "fmadd.ps   f7, f26, f18, f7\n"
         : 
         :
         : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7",
           "f18" , "f19", "f20", "f21", "f22", "f23", "f24",
           "f25"
        );
      }

      if (not Float16Dst) {
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

        dst_ptr += 64 * dstElemSize;
      }

      if (Float16Dst) {
        // Convert and store accumulated results.
        __asm__ __volatile__ (
          "fcvt.f16.ps f0, f0\n"
          "fcvt.f16.ps f1, f1\n"
          "fcvt.f16.ps f2, f2\n"
          "fcvt.f16.ps f3, f3\n"
          "fcvt.f16.ps f4, f4\n"
          "fcvt.f16.ps f5, f5\n"
          "fcvt.f16.ps f6, f6\n"
          "fcvt.f16.ps f7, f7\n"
          "fsch.ps f0, f29(%[dst_ptr])\n"
          "addi %[dst_ptr], %[dst_ptr], 16\n"
          "fsch.ps f1, f29(%[dst_ptr])\n"
          "addi %[dst_ptr], %[dst_ptr], 16\n"
          "fsch.ps f2, f29(%[dst_ptr])\n"
          "addi %[dst_ptr], %[dst_ptr], 16\n"
          "fsch.ps f3, f29(%[dst_ptr])\n"
          "addi %[dst_ptr], %[dst_ptr], 16\n"
          "fsch.ps f4, f29(%[dst_ptr])\n"
          "addi %[dst_ptr], %[dst_ptr], 16\n"
          "fsch.ps f5, f29(%[dst_ptr])\n"
          "addi %[dst_ptr], %[dst_ptr], 16\n"
          "fsch.ps f6, f29(%[dst_ptr])\n"
          "addi %[dst_ptr], %[dst_ptr], 16\n"
          "fsch.ps f7, f29(%[dst_ptr])\n"
          "addi %[dst_ptr], %[dst_ptr], 16\n"
          : [dst_ptr]   "+&r" (dst_ptr)
          :
          :
        );
      }

      minionCurrRowGroup++;

      minionCurrIndex += currSegmentLength;
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

      if (Float16Dst) {
        // Set offsets for storing float16 results (0, 2, 4, 6, 8, 10, 12, 14)
        __asm__ __volatile__ (
          "fadd.pi f29, f31, f31\n"
          :
          :
          : "f29"
        );
      }

      for (uintptr_t k = 0; k < (dstRowTailVRegs - 1); k++) {

        // For all sparse input rows.
        for (uintptr_t j = 0, currIndex = minionCurrIndex;
             j < currSegmentLength; j++, currIndex++) {
          int64_t            rowIndex   = indices[currIndex];
          volatile uint8_t * data_ptr   = tAInput + rowIndex * dataPitches[0];
          float            * scale_ptr  = (float *) &scales[rowIndex];
          float            * offset_ptr = (float *) &offsets[rowIndex];
          float            * weight_ptr = (float *) &tWInput[currIndex];

          __asm__ __volatile__ (
          	"fbc.ps  f26, 0x0(%[weight_ptr])\n"
            "fbc.ps  f27, 0x0(%[offset_ptr])\n"
            "fbc.ps  f28, 0x0(%[scale_ptr])\n"

            // Load a full input cache line (64 elements, 8 vregs)
            "fgb.ps     f25, f31, %[data_ptr]\n"
            "addi       %[data_ptr], %[data_ptr], 8\n"
           : [data_ptr]   "+&r" (data_ptr)
           : [weight_ptr] "r"   (weight_ptr),
             [offset_ptr] "r"   (offset_ptr),
             [scale_ptr]  "r"   (scale_ptr)
           : "f25", "f26", "f27", "f28"
          );

          if (Int8Src) {
            // Convert to UInt8_t adding 128.
            __asm__ __volatile__ (
              "faddi.pi f25, f25, 0x80\n"
             :
             : 
             : "f25"
            );
          }

          __asm__ __volatile__ (
            "fand.pi    f25, f25, f30\n"
            "fcvt.ps.pw f25, f25\n"
            "fmadd.ps   f25, f25, f28, f27\n"
            "fmadd.ps   f0, f26, f25, f0\n"
           : 
           :
           : "f0", "f25"
          );
        }

        if (not Float16Dst) {
          // Store accumulated results.
          __asm__ __volatile__ (
            "fsw.ps f0, (%[dst_ptr])\n"
            :
            : [dst_ptr] "r" (dst_ptr)
            :
          );
        }

        if (Float16Dst) {
          __asm__ __volatile__ (
            "fcvt.f16.ps f0, f0\n"
            "fsch.ps f0, f29(%[dst_ptr])\n"
            :
            : [dst_ptr] "r" (dst_ptr)
            :
          );
        }
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
        int64_t            rowIndex   = indices[currIndex];
        volatile uint8_t * data_ptr   = tAInput + rowIndex * dataPitches[0];
        float            * scale_ptr  = (float *) &scales[rowIndex];
        float            * offset_ptr = (float *) &offsets[rowIndex];
        float            * weight_ptr = (float *) &tWInput[currIndex];

        __asm__ __volatile__ (
          "fbc.ps  f26, 0x0(%[weight_ptr])\n"
          "fbc.ps  f27, 0x0(%[offset_ptr])\n"
          "fbc.ps  f28, 0x0(%[scale_ptr])\n"

          // Load a full input cache line (64 elements, 8 vregs)
          "fgb.ps     f25, f31, %[data_ptr]\n"
          "addi       %[data_ptr], %[data_ptr], 8\n"
         : [data_ptr]   "+&r" (data_ptr)
         : [weight_ptr] "r"   (weight_ptr),
           [offset_ptr] "r"   (offset_ptr),
           [scale_ptr]  "r"   (scale_ptr)
         : "f25", "f26", "f27", "f28"
        );

        if (Int8Src) {
          // Convert to UInt8_t adding 128.
        	__asm__ __volatile__ (
            "faddi.pi f25, f25, 0x80\n"
           :
           : 
           : "f25"
          );
        }

        __asm__ __volatile__ (
          "fand.pi    f25, f25, f30\n"
          "fcvt.ps.pw f25, f25\n"
          "fmadd.ps   f25, f25, f28, f27\n"
          "fmadd.ps   f0, f26, f25, f0\n"
         :
         :
         : "f0", "f25"
        );
      }

      if (not Float16Dst) {
        // Store accumulated results.
        __asm__ __volatile__ (
          "fsw.ps f0, (%[dst_ptr])\n"
          :
          : [dst_ptr] "r" (dst_ptr)
          :
        );
      }

      if (Float16Dst) {
        __asm__ __volatile__ (
          "fcvt.f16.ps f0, f0\n"
          "fsch.ps f0, f29(%[dst_ptr])\n"
          :
          : [dst_ptr] "r" (dst_ptr)
          :
        );
      }

      minionCurrIndex += currSegmentLength;

      // Move from row tail to next row.
      minionCurrSegment++;
      minionCurrRowGroup = 0;
      currSegmentLength = lengths[minionCurrSegment];

      dst_ptr = tOutput + (minionCurrSegment * dstPitches[0] + minionCurrRowGroup * 64) * dstElemSize;
    }
  }
}


GEN_INSTANCES_RQSLWS_V(template, fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstFloatTyVectorized,
            	void *pdst, void *pdstDims, void *pdstPitches, unsigned int pdstDimNum,
	    		void *pdata, void *pdataDims, void *pdataPitches, void *pscale,
	    		void *poffset, void *pweights, void *pweightsDims, void *pweightsPitches,
	    		void *pindices, void *plengths, unsigned int pLengthsSize, uint64_t flags,
	    		const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0);
