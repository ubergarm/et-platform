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

#ifndef _FUSED_ROWWISE_QUANTIZED_SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_
#define _FUSED_ROWWISE_QUANTIZED_SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_

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
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

#ifndef ACCUMULATOR_TYPE
#define ACCUMULATOR_TYPE

template<typename srcType>
struct accumulatorType {
  using type =
    typename std::conditional<std::is_same<srcType, int64_t>::value, int64_t,
      typename std::conditional<std::is_same<srcType, int32_t>::value, int32_t,
        float>::type >::type;
};

#endif

template <ElemKind elK>
inline __attribute((always_inline))
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst(
                   LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                   LibTensor* in3T, LibTensor* in4T,
                   uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  assert(elK == FloatTy || elK == Float16Ty);
  using dstType = typename elemKind2elemTy<elK>::type;

  if (get_minion_id() != minionOffset) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT-> dst in1T->data in2T->weight in3T->indices in4T->lengths */

  // float *tOutput = (float *)pdst;
  float *tOutput = outT->getRawDataPointer<dstType>();
  // uint8_t *tAInput = (uint8_t *)pdata;
  uint8_t *tAInput = in1T->getRawDataPointer<uint8_t>();
  // float *tWInput = (float *)pweights;
  float *tWInput = in2T->getRawDataPointer<float>();
  // long long *indices = (long long *)pindices;
  long long *indices = in3T->getRawDataPointer<long long>();
  // int32_t *lengths = (int32_t *)plengths;
  int32_t *lengths = in4T->getRawDataPointer<int32_t>();

  // unsigned int *dstIndex = (unsigned int *)pdstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dataIndex = (unsigned int *)pdataDims;
  const dim_t *dataIndex = in1T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *dataPitch = (unsigned int *)pdataPitches;
  const dim_t *dataPitch = in1T->strides().data();
  // unsigned int *weightPitch = (unsigned int *)pweightsPitches;
  const dim_t *weightPitch = in2T->strides().data();

  unsigned int pdstDimNum = static_cast<unsigned int>(outT->ndims());
  
  // size_t segments = pLengthsSize;
  const size_t segments = static_cast<size_t>(in4T->ndims());
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

  // Initialize to zero output tensor
  for (size_t i = 0; i < segments; i++) {
    size_t offsetOut = i * dstPitch[0];
    for (size_t k = 0; k < outLineSize; k++) {
      tOutput[offsetOut] = 0;
      offsetOut++;
    }
  }

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

template <ElemKind elK>
inline __attribute((always_inline))
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstThreaded(
                   LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                   LibTensor* in3T, LibTensor* in4T, uint64_t flags,
                   const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  assert(elK == FloatTy || elK == Float16Ty);
  using dstType = typename elemKind2elemTy<elK>::type;
  unsigned int minionId = get_minion_id();
  if (minionId < minionOffset) return;   // If Minion is outside the group assigned to this Node get out.
  minionId -= minionOffset;

  // Get number of Minions assigned to this Node.
  uint64_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT-> dst in1T->data in2T->weight in3T->indices in4T->lengths */

  // float *tOutput = (float *)pdst;
  const Addresser<elK> tOutputRd(outT->getRawDataPointer<void>(), outT->getScale(), outT->getOffset());
  Addresser<elK> tOutput(outT->getRawDataPointer<void>(), outT->getScale(), outT->getOffset());
  // uint8_t *tAInput = (uint8_t *)pdata;
  uint8_t *tAInput = in1T->getRawDataPointer<uint8_t>();
  // float *tWInput = (float *)pweights;
  float *tWInput = nullptr;
  if (in2T != nullptr) { tWInput = in2T->getRawDataPointer<float>(); }
  // long long *indices = (long long *)pindices;
  long long *indices = in3T->getRawDataPointer<long long>();
  // int32_t *lengths = (int32_t *)plengths;
  int32_t *lengths = in4T->getRawDataPointer<int32_t>();
  
  // unsigned int *dstIndex = (unsigned int *)pdstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dataIndex = (unsigned int *)pdataDims;
  const dim_t *dataIndex = in1T->dims().data();  

  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *dataPitch = (unsigned int *)pdataPitches;
  const dim_t *dataPitch = in1T->strides().data();
  // unsigned int *weightPitch = (unsigned int *)pweightsPitches;
  const dim_t *weightPitch = nullptr;
  if (in2T != nullptr) { weightPitch = in2T->strides().data(); }

  unsigned int pdstDimNum = static_cast<unsigned int>(outT->ndims());

  // size_t segments = pLengthsSize;
  const size_t segments = in4T->dims()[0];
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

  if (activeMinions > segments) {
    unsigned int minionsPerRow = activeMinions / segments;
    unsigned int rowId = minionId / minionsPerRow;
    unsigned int positionInRow = minionId - rowId * minionsPerRow;

    unsigned int total = (outLineSize - 1) / minionsPerRow + 1;
    unsigned int start = positionInRow * total;
    unsigned int end = start + total;
    if (end > outLineSize)
      end = outLineSize;
    unsigned int offsetOut = rowId * dstPitch[0];
    start += offsetOut;
    end += offsetOut;

    // Output tensor should be zero at the begin
    size_t idxStart = ranges[rowId];
    size_t idxEnd = idxStart + lengths[rowId];

    // For all the indices to compute a result
    for (size_t j = idxStart; j < idxEnd; j++) {
      const float weight = (tWInput != nullptr) ? tWInput[j * weightPitch[0]] : 1.0f;
      size_t offsetIn = indices[j] * dataPitch[0];

      // Get the scale and offset from the row; go to the current row and offset
      // into it up until the last 2 elments. Can't use size of as it doesn't work
      // correctly for float16.
      size_t sizeDstType;

      if (std::is_same<dstType, float>::value) { sizeDstType = 4; }
      else                                     { sizeDstType = 2; }

      uint8_t *currRowScaleOffsetPtr = &tAInput[offsetIn + inLineSize - (2 * sizeDstType)];

      float scale;
      float offset;
      // Regular load for float
      if (std::is_same<dstType, float>::value) {
        float * currRowScaleOffsetPtrFloat = (float *) currRowScaleOffsetPtr;
        scale  = currRowScaleOffsetPtrFloat[0];
        offset = currRowScaleOffsetPtrFloat[1];
      }
      // Upconvert to float for float16
      else {
        uint16_t * currRowScaleOffsetPtrUint16 = (uint16_t *) currRowScaleOffsetPtr;
        dnn_lib::convertFp16ToFp32(currRowScaleOffsetPtrUint16[0], scale);
        dnn_lib::convertFp16ToFp32(currRowScaleOffsetPtrUint16[1], offset);
      }

      offsetIn += positionInRow * total;

      // For all the elements of the row that the minion processes
      for (size_t k = start; k < end; k++) {
        float d = dequantizeWithFloatOffset(tAInput[offsetIn], scale, offset);
        typename accumulatorType<dstType>::type sum;

        // Need to reset to 0 if first accumulation
        if (j == idxStart) { sum = 0.0f; }
        else               { sum = tOutputRd[k]; }

        sum += d * weight;
        tOutput[k] = sum;
        offsetIn++;
      }
    }
  } else {

    unsigned int cll = CACHE_LINE_BYTES / sizeof(float);
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
        const float weight = (tWInput != nullptr) ? tWInput[curIdx * weightPitch[0]] : 1.0f;
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
        typename accumulatorType<dstType>::type sum = tOutputRd[offsetOut];
        for (size_t k = 0; k < outLineSize; k++) {

          float d = dequantizeWithFloatOffset(tAInput[offsetIn], scale, offset);
          sum += d * weight;
          offsetOut++;
          offsetIn++;
        }
        tOutput[offsetOut] = sum;
      }
    }
  }
}

//
// mask   m0  should be set to the proper mask for this vector
// vector f30 should be set to {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
// vector f31 should be set to {0, 1, 2, 3, 4, 5, 6, 7}
// vector f29 should be set to {0, 2, 4, 6, 8, 10, 12, 14}
//
  template <ElemKind elK>
inline __attribute((always_inline))
void fusedRowwiseQuantizedSparseLengthsWeightedSumVect(
    uintptr_t minionCurrIndex, uintptr_t currSegmentLength,
    uint8_t *tAInput, int64_t *indices, uintptr_t dataRowPitch,
    uintptr_t dataRowSize, uintptr_t dstElemSize,
    uint8_t *tWInput, uint8_t *dst_ptr, uint8_t *dst2_ptr, const bool Weighted = true) { 

  const bool float32Dst = elK == FloatTy;
  const bool float16Dst = elK == Float16Ty;

  // Clear vector accumulator at the start.
  __asm__ __volatile__ (
    "fxor.pi f0, f0, f0\n"
    : 
    :
    : "f0"
  );

  // For all sparse input rows.
  for (uintptr_t j = 0, currIndex = minionCurrIndex;
       j < currSegmentLength; j++, currIndex++) {
    uint8_t * data_ptr   = tAInput + indices[currIndex] * dataRowPitch;
    void             * scale_ptr  = (void *) &data_ptr[dataRowSize - dstElemSize * 2];
    void             * offset_ptr = (void *) &data_ptr[dataRowSize - dstElemSize    ];
  
    if (Weighted){
      uint8_t        * weight_ptr = &tWInput[currIndex * dstElemSize];
  
      __asm__ __volatile__ (
        "fbc.ps  f26, 0x0(%[weight_ptr])\n"
        :
        : [weight_ptr] "r" (weight_ptr)
        : "f26"
      );
  
      if (float16Dst) {
        __asm__ __volatile__ (
          "fcvt.ps.f16 f26, f26\n"
          :
          :
          : "f26"
        );
      }
    }
  
   __asm__ __volatile__ (
     "fbc.ps  f27, 0x0(%[offset_ptr])\n"
     "fbc.ps  f28, 0x0(%[scale_ptr])\n"
     :
     : [offset_ptr] "r"   (offset_ptr),
       [scale_ptr]  "r"   (scale_ptr)
     : "f27", "f28"
   );
  
   if (float16Dst) {
     __asm__ __volatile__ (
       "fcvt.ps.f16 f27, f27\n"
       "fcvt.ps.f16 f28, f28\n"
       :
       :
       : "f27", "f28"
     );
   }
  
   __asm__ __volatile__ (
      // Load a full input cache line (64 elements, 8 vregs)
      "fgb.ps     f25, f31, %[data_ptr]\n"
      "addi       %[data_ptr], %[data_ptr], 8\n"
      "fand.pi    f25, f25, f30\n"
      "fcvt.ps.pw f25, f25\n"
      "fmadd.ps   f25, f25, f28, f27\n"
     : [data_ptr]   "+&r" (data_ptr)
     : [offset_ptr] "r"   (offset_ptr),
       [scale_ptr]  "r"   (scale_ptr)
     : "f25"
    );
    if (Weighted) {
      __asm__ __volatile__ (
        "fmadd.ps f0, f26, f25, f0\n"
        :
        :
        : "f0"
      );
    }
    else {
      __asm__ __volatile__ (
        "fadd.ps f0, f25, f0\n"
        :
        :
        : "f0"
      );
    }
  }
  
  if (float32Dst) {
    // Store accumulated results.
    __asm__ __volatile__ (
      "fsw.ps f0, (%[dst_ptr])\n"
      :
      : [dst_ptr] "r" (dst_ptr)
      :
    );
  }
  
  if ((not float32Dst) and float16Dst) {
    __asm__ __volatile__ (
      "fcvt.f16.ps f0, f0\n"
      "fsch.ps f0, f29(%[dst_ptr])\n"
      :
      : [dst_ptr] "r" (dst_ptr)
      : "f0"
    );
  }

  dst_ptr  += 8 * dstElemSize;
  
  if (float32Dst and float16Dst) {
    __asm__ __volatile__ (
      "fcvt.f16.ps f0, f0\n"
      "fsch.ps f0, f29(%[dst_ptr])\n"
      :
      : [dst_ptr] "r" (dst2_ptr)
      : "f0"
    );

    dst2_ptr += 8 * 2;
  }
}

 
template <ElemKind elK, bool Weighted = true>
inline __attribute((always_inline))
void fusedRowwiseQuantizedSparseLengthsWeightedSumInstVectorizedImpl(
        LibTensor* outT, LibTensor* out2T, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
        uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
    
  const bool float32Dst = elK == FloatTy;
  const bool float16Dst = elK == Float16Ty;

  assert(elK == FloatTy || elK == Float16Ty);
  // Get offset of the Minion inside the group of Minions assigned to this Node.

  uint64_t minionId = get_minion_id();
  if (minionId < minionOffset) return;   // If Minion is outside the group assigned to this Node get out.
  minionId -= minionOffset;
  // Get number of Minions assigned to this Node.
  uint64_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;

  // If Minion is outside the group assigned to this Node get out.
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* outT--> dest out2T --> dest2 in1T--> data in2T-->weight in3T-->indices in4T-->length*/
  
  // Set real types for input pointers.
  // For dst we used uint8_t because it can be accessed with different types.
  // uint8_t *tOutput = (uint8_t *) pdst;
  uint8_t *tOutput = outT->getRawDataPointer<uint8_t>();
  // uint8_t *tAInput = (uint8_t *) pdata;
  uint8_t *tAInput = in1T->getRawDataPointer<uint8_t>();
  // uint8_t *tWInput = (uint8_t *) pweights;
  uint8_t *tWInput = in2T->getRawDataPointer<uint8_t>();
  // int64_t *indices = (int64_t *) pindices;
  int64_t *indices = in3T->getRawDataPointer<int64_t>();
  // int32_t *lengths = (int32_t *) plengths;
  int32_t *lengths = in4T->getRawDataPointer<int32_t>();

  // uint32_t *dstDims     = (uint32_t *) pdstDims;
  const dim_t *dstDims = outT->dims().data();
  // uint32_t *dataDims    = (uint32_t *) pdataDims;
  const dim_t *dataDims = in1T->dims().data();
  // uint32_t *dstPitches  = (uint32_t *) pdstPitches;
  const dim_t *dstPitches = outT->strides().data();
  // uint32_t *dataPitches = (uint32_t *) pdataPitches;
  const dim_t *dataPitches = in1T->strides().data();

  unsigned int pdstDimNum = static_cast<unsigned int>(outT->ndims());
  
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
  //uintptr_t dataRowSize = 1;
  //for (uintptr_t i = 1; i < pdstDimNum; i++) dataRowSize *= dataDims[i];
  // The data tensor must have only two dimensions.
  uintptr_t dataRowSize = dataDims[1];

  // Compute the number of elements per output row (first tensor dimension).
  uintptr_t dstRowSize = 1;
  for (uintptr_t i = 1; i < pdstDimNum; i++) dstRowSize *= dstDims[i];

  // Get size of the output element.
  uintptr_t dstElemSize;
  if (float32Dst)  // For dual output use float32 blocking for the tail
    dstElemSize = 4;
  else if (float16Dst)
    dstElemSize = 2;

  // Compute the number of 8-element vectors per output cache line.
  uintptr_t dstCacheLineVRegs = CACHE_LINE_BYTES / (dstElemSize * 8);

  // Compute the number of Cache Line groups per output row (rounded up).
  uintptr_t dstRowGroups = ((dstRowSize - 1) / CACHE_LINE_BYTES) + 1;

  // Determine if row has a tail.
  bool dstRowHasTail = ((dstRowSize % CACHE_LINE_BYTES) != 0);

  // Compute the number of 8-element vectors in the tail of the row.
  uintptr_t dstRowTailVRegs = (((dstRowSize - 1) / 8) + 1) % dstCacheLineVRegs;

  // Compute the element mask for the tail of the row.
  uint8_t dstRowTailVRegMask = (1 << (((dstRowSize - 1) % 8) + 1)) - 1;

  // Assign work to Minions :
  //
  // - Each Minion gets assigned at least 64 output elements or a full output row
  //   if the row dimension is smaller than 64.
  //

  uintptr_t totalWorkUnits = dstRowGroups * dstDims[0];

  //  Distribute the tail of groups.
  uintptr_t minionWorkUnits = 0;
  uintptr_t minionFirstWorkUnit = 0;

  if ((totalWorkUnits % activeMinions) == 0) {
    minionWorkUnits = totalWorkUnits / activeMinions;
    minionFirstWorkUnit = minionId * minionWorkUnits;
  } else {
    minionWorkUnits = totalWorkUnits / activeMinions;
    uintptr_t remainingWorkUnits = totalWorkUnits % activeMinions;
    if (minionId < remainingWorkUnits) {
      minionWorkUnits++;
      // Compute the index into the first work unit.
      minionFirstWorkUnit = minionId * minionWorkUnits;
    } else {
      // Compute the index into the first work unit.
      minionFirstWorkUnit = remainingWorkUnits * (minionWorkUnits + 1)
                         +  (minionId - remainingWorkUnits) * minionWorkUnits;
    }
  }

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

  // Second output pointer when both float32 and float16 are active.
  uint8_t *dst2_ptr;
  if (float32Dst and float16Dst)
    dst2_ptr = tOutput + (minionCurrSegment * dstPitches[0] + minionCurrRowGroup * 64) * 2;

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

      if (float16Dst) {
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
        volatile uint8_t * data_ptr   = tAInput + indices[currIndex] * dataPitches[0];
        void             * scale_ptr  = (void *) &data_ptr[dataRowSize - dstElemSize * 2];
        void             * offset_ptr = (void *) &data_ptr[dataRowSize - dstElemSize    ];

        if (Weighted){
          uint8_t        * weight_ptr = &tWInput[currIndex * dstElemSize];

          __asm__ __volatile__ (
            "fbc.ps  f26, 0x0(%[weight_ptr])\n"
            :
            : [weight_ptr] "r" (weight_ptr)
            : "f26"
          );

          if (float16Dst) {
            __asm__ __volatile__ (
              "fcvt.ps.f16 f26, f26\n"
              :
              :
              : "f26"
            );
          }
        }

        __asm__ __volatile__ (
          "fbc.ps  f27, 0x0(%[offset_ptr])\n"
          "fbc.ps  f28, 0x0(%[scale_ptr])\n"
          :
          : [offset_ptr] "r" (offset_ptr),
            [scale_ptr]  "r" (scale_ptr)
          : "f27", "f28"
        );

        if (float16Dst) {
          __asm__ __volatile__ (
            "fgb.ps      f25, f31, %[data_ptr]\n"
            "fcvt.ps.f16 f27, f27\n"
            "fcvt.ps.f16 f28, f28\n"
            :
            : [data_ptr] "r" (data_ptr)
            : "f25", "f27", "f28"
          );
        }
        else {
          __asm__ __volatile__ (
            "fgb.ps f25, f31, %[data_ptr]\n"
            :
            : [data_ptr] "r" (data_ptr)
            :  "f25"
          );
        }

        __asm__ __volatile__ (
          // Load a full input cache line (64 elements, 8 vregs)
          //
          // NOTE: Moved first gather of data tensor before the
          // converts as an optimization.
          //
          //"fgb.ps     f25, f31, %[data_ptr]\n"
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
         : [data_ptr]   "+&r" (data_ptr)
         : [offset_ptr] "r"   (offset_ptr),
           [scale_ptr]  "r"   (scale_ptr)
         : "f18", "f19", "f20", "f21", "f22",
           "f23", "f24", "f25", "f27", "f28"
        );

        if (Weighted) {
          __asm__ __volatile__ (
            "fmadd.ps f0, f26, f25, f0\n"
            "fmadd.ps f1, f26, f24, f1\n"
            "fmadd.ps f2, f26, f23, f2\n"
            "fmadd.ps f3, f26, f22, f3\n"
            "fmadd.ps f4, f26, f21, f4\n"
            "fmadd.ps f5, f26, f20, f5\n"
            "fmadd.ps f6, f26, f19, f6\n"
            "fmadd.ps f7, f26, f18, f7\n"
            :
            :
            : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"
          );
        }
        else {
          __asm__ __volatile__ (
            "fadd.ps f0, f25, f0\n"
            "fadd.ps f1, f24, f1\n"
            "fadd.ps f2, f23, f2\n"
            "fadd.ps f3, f22, f3\n"
            "fadd.ps f4, f21, f4\n"
            "fadd.ps f5, f20, f5\n"
            "fadd.ps f6, f19, f6\n"
            "fadd.ps f7, f18, f7\n"
            :
            :
            : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"
          );
        }
      }

      if (float32Dst) {
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

      if ((not float32Dst) and (float16Dst)) {
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
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"
        );
      }

      if (float32Dst and (float16Dst)) {
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
          : [dst_ptr]   "+&r" (dst2_ptr)
          : 
          : "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"
        );
      }

      minionCurrIndex += currSegmentLength;

      if (minionCurrRowGroup != (dstRowGroups - 1)) {
        minionCurrRowGroup++;
      } else {
        // Move from row tail to next row.
        minionCurrSegment++;
        minionCurrRowGroup = 0;
        currSegmentLength = lengths[minionCurrSegment];

        dst_ptr = tOutput + (minionCurrSegment * dstPitches[0] + minionCurrRowGroup * 64) * dstElemSize;
        if (float32Dst and float16Dst)
          dst2_ptr = tOutput + (minionCurrSegment * dstPitches[0] + minionCurrRowGroup * 64) * 2;
      }
    }
    else {
      volatile int32_t gather_offsets[] = { 0, 1, 2, 3, 4, 5, 6, 7 };

      // Initialize mask to clear upper bytes from input load
      __asm__ __volatile__ (
        "mov.m.x m0, zero, 0xff\n"
      );

      // Initialize offsets for gather from input
      __asm__ __volatile__ (
        "li      t0, 0xff\n"
        "fbcx.ps f30, t0\n"
        "flw.ps  f31, 0x0(%[gather_offsets])\n"
        :
        : [gather_offsets] "r" (gather_offsets)
        : "t0", "f0", "f30", "f31"
      );

      if (float16Dst) {
        // Set offsets for storing float16 results (0, 2, 4, 6, 8, 10, 12, 14)
        __asm__ __volatile__ (
          "fadd.pi f29, f31, f31\n"
          :
          :
          : "f29"
        );
      }

      for (uintptr_t k = 0; k < (dstRowTailVRegs - 1); k++) {
          fusedRowwiseQuantizedSparseLengthsWeightedSumVect<elK>(
          minionCurrIndex, currSegmentLength, tAInput, indices,
          dataPitches[0], dataRowSize, dstElemSize,
          tWInput, dst_ptr, dst2_ptr, Weighted);
      }

      // Set mask for last VReg in group.
      __asm__ __volatile__ (
        "mov.m.x m0, %[tail_mask], 0x0\n"
        :
        : [tail_mask] "r" (dstRowTailVRegMask)
      );

      fusedRowwiseQuantizedSparseLengthsWeightedSumVect<elK>(
        minionCurrIndex, currSegmentLength, tAInput, indices,
        dataPitches[0], dataRowSize, dstElemSize,
        tWInput, dst_ptr, dst2_ptr, Weighted);

      minionCurrIndex += currSegmentLength;

      // Move from row tail to next row.
      minionCurrSegment++;
      minionCurrRowGroup = 0;
      currSegmentLength = lengths[minionCurrSegment];

      dst_ptr = tOutput + (minionCurrSegment * dstPitches[0] + minionCurrRowGroup * 64) * dstElemSize;
      if (float32Dst and float16Dst)
        dst2_ptr = tOutput + (minionCurrSegment * dstPitches[0] + minionCurrRowGroup * 64) * 2;
    }
  }
}

template <ElemKind elK>
inline __attribute((always_inline))
void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstVectorized(
       LibTensor* outT, LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, LibTensor* in4T,
       uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  inlining::fusedRowwiseQuantizedSparseLengthsWeightedSumInstVectorizedImpl<elK, true>
    (outT, nullptr, in1T, in2T, in3T, in4T, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _FUSED_ROWWISE_QUANTIZED_SPARSE_LENGTHS_WEIGHTED_SUM_INST_H_
