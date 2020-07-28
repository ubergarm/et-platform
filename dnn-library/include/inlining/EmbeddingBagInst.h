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

#ifndef _EMBEDDING_BAG_INST_H_
#define _EMBEDDING_BAG_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "Float16.h"
#include "utils.h" // From include/internal path
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief Convert a length vector to a range sequence. 
 *
 * For example, input=[4,3,1], the output would be [0,1,2,3,0,1,2,0].
 *
 * Currently It only solves Int32ITy ElemKind following InstGen.cpp
 * Interpreter.cpp and isOpSupported at ETSOC.cpp specification.
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected data.
 * @param[in] in1T LibTensor input. It keeps the Data to being handle.
 * @param[in] in2T LibTensor input. It keeps the Weights to being handle.
 * @param[in] in3T LibTensor input. It keeps the indices to being handle.
 * @param[in] in4T LibTensor input. It keeps the offsets to being handle.
 * @param[in] hasEndOffset bool type mark the end of the last segment.
 * @param[flags] flags Gives the information of the Active Shires and the
 *  type of evict required.
 */
template <ElemKind elKind>
inline typename std::enable_if_t<(elKind == Float16Ty), void>
fwdLibEmbeddingBagInst(LibTensor* outT, LibTensor *in1T, LibTensor* in2T, 
                       LibTensor* in3T, LibTensor* in4T, bool hasEndOffset,
                       uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  
  if (get_minion_id() != minionOffset) return;

  assert(in1T->getElementType() == outT->getElementType());
  assert(in1T->getElementType() == Float16Ty);
  assert((in3T->getElementType() == Int64ITy) && (in3T->getElementType() == in4T->getElementType()));

  using elkType = typename elemKind2elemTy<elKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = in1T->getHandle<elkType>();
  auto weightH = in2T->getHandle<elkType>();
  auto indxH = in3T->getHandle<int64_t>();
  auto offH = in4T->getHandle<int64_t>();

  outH.zero();

  const dim_t segments = hasEndOffset ? (in4T->dims()[0] - 1) : in4T->dims()[0];
  const dim_t numIndices = in3T->dims()[0];

/*   // NOTE : Pitch is passed as the number of elements, not as bytes. */
/*   const size_t lineSize = dataDim1Pitch; */
/*   //@TODO in SW-2429 remove dataDim1Pitch param once the instruction bellow It works.  */
/*   //const dim_t lineSize = (in1T->strides().data()[0]/in1T->getElementSize()); */

  const dim_t lineSize = in1T->strides()[0];
  const dim_t outLineSize = outT->strides()[0];
  //dim_t lineSize = in1T->actualSize() / in1T->getElementSize();

  dim_t curIdx = 0;
  for (dim_t i = 0; i < segments; i++) {
    dim_t start = offH.raw(i);
    dim_t end;
    if(!hasEndOffset) {
      // Note that in this case we have to use numIndices to find the end of
      // the last segment. This is an issue though because it relies on knowing
      // the total length of the indices tensor which may not be possible.
      // Future implementations of this operator should always give an end
      // offset so eventually this case should be removed.
      end = (i == (segments-1))? numIndices : offH.raw(i + 1);
    }
    else {
      end = offH.raw(i + 1);
    }

    if (start == end) {
      continue;
    }
    else if (start > end) {
      break;
    }

    for (dim_t j = start; j < end; j++) {

      float weightfl;
      convertFp16ToFp32(static_cast<uint16_t>(weightH.raw(curIdx)), weightfl);
      dim_t offsetIn = indxH.raw(curIdx++) * lineSize;
      dim_t offsetOut = i * outLineSize;
      for (dim_t k = 0; k < lineSize; k++) {
        float datafl = 0;
        float outfl = 0;
        uint16_t out16 = 0;
        convertFp16ToFp32(static_cast<uint16_t>(dataH.raw(offsetIn++)), datafl);
        convertFp16ToFp32(static_cast<uint16_t>(outH.raw(offsetOut)), outfl);
        outfl += datafl * weightfl;
        convertFp32ToFp16(outfl,out16);
        outH.raw(offsetOut++) = out16;
      }
    }
  }

  outT->evict(DO_EVICTS);
}

template <ElemKind elKind>
inline typename std::enable_if_t<(elKind == FloatTy), void>
fwdLibEmbeddingBagInst(LibTensor* outT, LibTensor *in1T, LibTensor* in2T, 
                       LibTensor* in3T, LibTensor* in4T, bool hasEndOffset,
                       uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(in1T->getElementType() == outT->getElementType());
  assert(in1T->getElementType() == FloatTy);
  assert((in3T->getElementType() == Int64ITy) && (in3T->getElementType() == in4T->getElementType()));

  using elkType = typename elemKind2elemTy<elKind>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = in1T->getHandle<elkType>();
  auto weightH = in2T->getHandle<elkType>();
  auto indxH = in3T->getHandle<int64_t>();
  auto offH = in4T->getHandle<int64_t>();

  outH.zero();

  const dim_t segments = hasEndOffset ? (in4T->dims()[0] - 1) : in4T->dims()[0];
  const dim_t numIndices = in3T->dims()[0];

/*   // NOTE : Pitch is passed as the number of elements, not as bytes. */
/*   const size_t lineSize = dataDim1Pitch; */
/*   //@TODO in SW-2429 remove dataDim1Pitch param once the instruction bellow It works.  */
/*   //const dim_t lineSize = (in1T->strides().data()[0]/in1T->getElementSize()); */
  const dim_t lineSize = in1T->strides()[0];
  const dim_t outLineSize = outT->strides()[0];
  //dim_t lineSize = in1T->actualSize() / in1T->getElementSize();

  dim_t curIdx = 0;
  for (dim_t i = 0; i < segments; i++) {
    dim_t start = offH.raw(i);
    dim_t end;
    if(!hasEndOffset) {
      // Note that in this case we have to use numIndices to find the end of
      // the last segment. This is an issue though because it relies on knowing
      // the total length of the indices tensor which may not be possible.
      // Future implementations of this operator should always give an end
      // offset so eventually this case should be removed.
      end = (i == (segments-1))? numIndices : offH.raw(i + 1);
    }
    else {
      end = offH.raw(i + 1);
    }

    if (start == end) {
      continue;
    }
    else if (start > end) {
      break;
    }

    for (dim_t j = start; j < end; j++) {
      elkType weight = weightH.raw(curIdx);      
      dim_t offsetIn = indxH.raw(curIdx++) * lineSize;
      dim_t offsetOut = i * outLineSize;
      for (dim_t k = 0; k < lineSize; k++) {
        outH.raw(offsetOut++) += dataH.raw(offsetIn++) * weight;
      }
    }
  }

  outT->evict(DO_EVICTS);
}

template <ElemKind elK>
inline __attribute((always_inline))
void embeddingBagsTailVectorized(
    uintptr_t minionCurrIndex, uintptr_t currSegmentStart,
    uintptr_t currSegmentEnd,
    uint8_t *tAInput, int64_t *indices, uintptr_t dataRowPitch,
    uint8_t *tWInput, uint8_t *dst_ptr) { 

  const bool float32Dst = (elK == FloatTy);
  const bool float16Dst = (elK == Float16Ty);

  assert(elK == FloatTy || elK == Float16Ty);

  const uintptr_t elemSize = float32Dst ? 4 : 2;

  // Clear vector accumulator at the start.
  __asm__ __volatile__ (
    "fxor.pi f0, f0, f0\n"
    : 
    :
    : "f0"
  );

  // For all sparse input rows.
  for (uintptr_t j = currSegmentStart, currIndex = minionCurrIndex;
       j < currSegmentEnd; j++, currIndex++) {

    uint8_t *data_ptr   = tAInput + indices[currIndex] * dataRowPitch;
    uint8_t *weight_ptr = tWInput + currIndex * elemSize;
  
    __asm__ __volatile__ (
      "fbc.ps  f21, 0x0(%[weight_ptr])\n"
      :
      : [weight_ptr] "r" (weight_ptr)
      : "f21"
    );
  
    if (float16Dst) {
      __asm__ __volatile__ (
        "fcvt.ps.f16 f21, f21\n"
        :
        :
        : "f21"
      );
    }
    
    if (float32Dst) { 
      __asm__ __volatile__ (
        "flw.ps     f10, (%[data_ptr])\n"
        :
        : [data_ptr] "r" (data_ptr)
        : "f10"
      );
      data_ptr += 32;
    } else {
      __asm__ __volatile__ (
        "fgh.ps      f10, f20, %[data_ptr]\n"
        "fcvt.ps.f16 f10, f10\n"
        :
        : [data_ptr] "r" (data_ptr)
        : "f10"
      );
    }

    __asm__ __volatile__ (
      "fmadd.ps f0, f21, f10, f0\n"
      :
      :
      : "f0"
    );
  }
  
  if (float32Dst) {
    // Store accumulated results.
    __asm__ __volatile__ (
      "fsw.ps f0, (%[dst_ptr])\n"
      :
      : [dst_ptr] "r" (dst_ptr)
      :
    );
  } else {
    __asm__ __volatile__ (
      "fcvt.f16.ps f0, f0\n"
      "fsch.ps f0, f20(%[dst_ptr])\n"
      :
      : [dst_ptr] "r" (dst_ptr)
      : "f0"
    );
  }

  dst_ptr  += 8 * elemSize;
}



template <ElemKind elK>
inline __attribute((always_inline))
void fwdLibEmbeddingBagInstVectorized(LibTensor* outT, LibTensor *in1T, LibTensor* in2T, 
                                 LibTensor* in3T, LibTensor* in4T, bool hasEndOffset,
                                 uint64_t flags, const uint32_t minionOffset = 0,
                                 const uint32_t assignedMinions = 0) {

  const bool float32Dst = (elK == FloatTy);
  const bool float16Dst = (elK == Float16Ty);

  assert(elK == FloatTy || elK == Float16Ty);

  // Get first ID for the first minion assigned to this operation.
  uint64_t minionId = get_minion_id();

  // If Minion is outside the group assigned to this Node get out.
  if (minionId < minionOffset) return;

  // Rebase minion ID.
  minionId -= minionOffset;

  // Get number of Minions assigned to this Node.
  uint64_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;

  // If Minion is outside the group assigned to this Node get out.
  if (minionId >= activeMinions) return;

  assert(in1T->getElementType() == outT->getElementType());
  assert(in1T->getElementType() == FloatTy);
  assert((in3T->getElementType() == Int64ITy) && (in3T->getElementType() == in4T->getElementType()));

  //using elkType = typename elemKind2elemTy<elK>::type;

  //auto outH = outT->getHandle<elkType>();
  //auto dataH = in1T->getHandle<elkType>();
  //auto weightH = in2T->getHandle<elkType>();
  //auto indxH = in3T->getHandle<int64_t>();
  auto offH = in4T->getHandle<int64_t>();

  auto tOutput = outT->getRawDataPointer<uint8_t>();
  auto tAInput = in1T->getRawDataPointer<uint8_t>();
  auto tWInput = in2T->getRawDataPointer<uint8_t>();
  auto indices = in3T->getRawDataPointer<int64_t>();
  //auto offsets = in4T->getRawDataPointer<int64_t>();

  const dim_t segments = hasEndOffset ? (in4T->dims()[0] - 1) : in4T->dims()[0];
  const dim_t numIndices = in3T->dims()[0];

  // Compute the number of elements per data row (first tensor dimension).
  const uintptr_t dstRowSize = outT->dims()[1];  
  
  // Get size of the output element.
  const uintptr_t elemSize = float32Dst ? 4 : 2;

  // Assign work to Minions :
  //
  // Each Minion gets assigned at least an output cache line to avoid
  // coherence issues when writing..
  // 
  // NOTE: NOT IMPLEMENTED!!!
  //
  // If the row is smaller than a cache line then multiple rows need to
  // be assigned to a single Minion.
  //

  // Compute the number of 8-element vectors (VRegs) per output cache line
  // (a Minion must be assigned a full cache line to avoid coherence issues
  // when writing).
  // This correspond with a group (one or more VRegs).
  const uintptr_t dstVRegElems = 8;  // SIMD/Vector length 
  const uintptr_t dstGroupElems = CACHE_LINE_BYTES /  elemSize;
  const uintptr_t dstGroupVRegs = CACHE_LINE_BYTES / (elemSize * 8);

  // Compute the number of groups per output row (rounded up).
  const uintptr_t dstRowGroups = ((dstRowSize - 1) / dstGroupElems) + 1;

  // Compute the number of 8-element vectors in the tail of the row
  // (number of VRegs not multiple of Group VRegs).
  const uintptr_t dstRowTailVRegs = (dstRowSize % dstVRegElems) % dstGroupVRegs;

  // Determine if row has a tail
  const bool dstRowHasTail = (dstRowTailVRegs != 0);

  // Compute the element mask for the last VReg in the row.
  const uint8_t dstRowTailVRegMask = (1 << (dstRowSize % dstVRegElems)) - 1;

  uintptr_t totalWorkUnits = dstRowGroups * outT->dims()[0];

  //  Distribute the tail of groups.
  uintptr_t minionWorkUnits = 0;
  uintptr_t minionFirstWorkUnit = 0;

  if ((totalWorkUnits % activeMinions) == 0) {
    minionWorkUnits = totalWorkUnits / activeMinions;
    minionFirstWorkUnit = minionId * minionWorkUnits;
  } else {
    minionWorkUnits = totalWorkUnits / activeMinions;
    const uintptr_t remainingWorkUnits = totalWorkUnits % activeMinions;
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

  // No work for this Minion.
  if (minionWorkUnits == 0)
    return;

  // Compute the first output row (segment) assigned to the Minion.
  uintptr_t minionFirstSegment = minionFirstWorkUnit / dstRowGroups;

  // Compute current group in row assigned to the Minion.
  uintptr_t minionFirstRowGroup = minionFirstWorkUnit % dstRowGroups;

  // Get the first index assigned to the Minion.
  uintptr_t minionFirstIndex = 0;
  uintptr_t currSegmentStart = 0;
  uintptr_t currSegmentEnd   = 0;

  auto getNextSegment = [&](uintptr_t segment) {
    bool emptySegment;
    dim_t start = offH.raw(segment);
    dim_t end;
    if (!hasEndOffset) {
      // Note that in this case we have to use numIndices to find the end of
      // the last segment. This is an issue though because it relies on knowing
      // the total length of the indices tensor which may not be possible.
      // Future implementations of this operator should always give an end
      // offset so eventually this case should be removed.
      end = (segment == (segments - 1)) ? numIndices : offH.raw(segment + 1);
    } else {
      end = offH.raw(segment + 1);
    }
    if (start == end) {
      emptySegment = true;
    } else if (start > end) {
      emptySegment = true;
    } else {
      emptySegment = false;
    }
    currSegmentStart = start;
    currSegmentEnd   = end;
    return emptySegment;
  };
  
  bool minionEmptySegment = false;
  for (uintptr_t s = 0; s <= minionFirstSegment; s++) {
    if (not minionEmptySegment)
      minionFirstIndex += (currSegmentEnd - currSegmentStart);
    minionEmptySegment = getNextSegment(s);
  }

  // Initialize indices.
  uintptr_t minionCurrIndex    = minionFirstIndex;
  uintptr_t minionCurrSegment  = minionFirstSegment;
  uintptr_t minionCurrRowGroup = minionFirstRowGroup;

  // Initilize output pointer.
  auto dst_ptr = tOutput + minionCurrSegment * outT->strides()[0] * elemSize + minionCurrRowGroup * 64;

  // For all minion assigned work units
  for (uintptr_t i = 0; i < minionWorkUnits; i++) {

    // Detect row tail
    bool dstGroupNotInRowTail = !dstRowHasTail || (minionCurrRowGroup != (dstRowGroups - 1));

    if (dstGroupNotInRowTail) {
      // Not in tail
      volatile int32_t gather_offsets16[] = { 0, 2, 4, 6, 8, 10, 12, 14 };

      // Initialize vector mask
      // Clear vector registers that will be used for accumulation
      // Initialize offsets for gather from input
      // Initialize mask to clear upper bytes from input load
      __asm__ __volatile__ (
        "mov.m.x m0, zero, 0xff\n"
        "fxor.pi f0, f0, f0\n"
        "fxor.pi f1, f0, f0\n"
        :
        :
        : "f0", "f1"
      );
      
      if (float16Dst) {
        __asm__ __volatile__ (
          "fxor.pi f2, f0, f0\n"
          "fxor.pi f3, f0, f0\n"
          "flw.ps  f20, 0x0(%[gather_offsets16])\n"
          :
          : [gather_offsets16] "r" (gather_offsets16)
          : "f2", "f3", "f20"
       );
      }

      // For all sparse input rows.
      for (uintptr_t j = currSegmentStart, currIndex = minionCurrIndex;
           j < currSegmentEnd; j++, currIndex++) {
        volatile uint8_t *data_ptr   = tAInput + indices[currIndex] * in1T->strides()[0] * elemSize;
        volatile uint8_t *weight_ptr = tWInput + currIndex * elemSize;

        __asm__ __volatile__ (
          "fbc.ps  f21, 0x0(%[weight_ptr])\n"
          :
          : [weight_ptr] "r" (weight_ptr)
          : "f21"
        );

        if (float16Dst) {
          __asm__ __volatile__ (
            "fcvt.ps.f16 f21, f21\n"
            :
            :
            : "f21"
          );
        }
        
        if (float16Dst) {
          __asm__ __volatile__ (
            "fgh.ps      f10, f20, %[data_ptr]\n"
            "addi        %[data_ptr], %[data_ptr], 16\n"
            "fgh.ps      f11, f20, %[data_ptr]\n"
            "addi        %[data_ptr], %[data_ptr], 16\n"
            "fgh.ps      f12, f20, %[data_ptr]\n"
            "addi        %[data_ptr], %[data_ptr], 16\n"
            "fgh.ps      f13, f20, %[data_ptr]\n"
            "addi        %[data_ptr], %[data_ptr], 16\n"
            "fcvt.ps.f16 f10, f10\n"
            "fcvt.ps.f16 f11, f11\n"
            "fcvt.ps.f16 f12, f12\n"
            "fcvt.ps.f16 f13, f13\n"
            "fmadd.ps    f0, f21, f10, f0\n"
            "fmadd.ps    f1, f21, f11, f1\n"
            "fmadd.ps    f2, f21, f12, f2\n"
            "fmadd.ps    f3, f21, f13, f3\n"
            :
            : [data_ptr] "r" (data_ptr)
            : "f0",  "f1",  "f2",  "f3",
              "f10", "f11", "f12", "f13"
          );
        } else {    // Float32
          __asm__ __volatile__ (
            "flw.ps   f10,   (%[data_ptr])\n"
            "flw.ps   f11, 32(%[data_ptr])\n"
            "fmadd.ps f0, f21, f10, f0\n"
            "fmadd.ps f1, f21, f11, f1\n"
            :
            : [data_ptr] "r" (data_ptr)
            : "f0", "f1", "f10", "f11"
          );
        }
      }

      if (float32Dst) {
        // Store accumulated results.
        __asm__ __volatile__ (
          "fsw.ps f0,    (%[dst_ptr])\n"
          "fsw.ps f1,  32(%[dst_ptr])\n"
          :
          : [dst_ptr] "r" (dst_ptr)
          :
        );

        dst_ptr += 64;
      } else { // Float16
        // Convert and store accumulated results.
        __asm__ __volatile__ (
          "fcvt.f16.ps f0, f0\n"
          "fcvt.f16.ps f1, f1\n"
          "fcvt.f16.ps f2, f2\n"
          "fcvt.f16.ps f3, f3\n"
          "fsch.ps     f0, f20(%[dst_ptr])\n"
          "addi        %[dst_ptr], %[dst_ptr], 16\n"
          "fsch.ps     f1, f20(%[dst_ptr])\n"
          "addi        %[dst_ptr], %[dst_ptr], 16\n"
          "fsch.ps     f2, f20(%[dst_ptr])\n"
          "addi        %[dst_ptr], %[dst_ptr], 16\n"
          "fsch.ps     f3, f20(%[dst_ptr])\n"
          "addi        %[dst_ptr], %[dst_ptr], 16\n"
          : [dst_ptr] "+&r" (dst_ptr)
          :
          : "f0", "f1", "f2", "f3"
        );
      }

      minionCurrIndex += (currSegmentEnd - currSegmentStart);

      if (minionCurrRowGroup != (dstRowGroups - 1)) {
        minionCurrRowGroup++;
      } else {
        // Move from row tail to next row.
        minionCurrSegment++;
        minionCurrRowGroup = 0;

        getNextSegment(minionCurrSegment);

        dst_ptr = tOutput + minionCurrSegment * outT->strides()[0] * elemSize + minionCurrRowGroup * 64;
      }
    } else {
      volatile int32_t gather_offsets16[] = { 0, 2, 4,  6,  8, 10, 12, 14 };

      // Initialize mask to clear upper bytes from input load
      __asm__ __volatile__ (
        "mov.m.x m0, zero, 0xff\n"
      );

      if (float16Dst) {
        // Initialize offsets for gather from input
        __asm__ __volatile__ (
          "flw.ps  f20, 0x0(%[gather_offsets16])\n"
          :
          : [gather_offsets16] "r" (gather_offsets16)
          : "f20"
        );
      }

      for (uintptr_t k = 0; k < (dstRowTailVRegs - 1); k++) {
        embeddingBagsTailVectorized<elK>(minionCurrIndex,
          currSegmentStart, currSegmentEnd,
          tAInput, indices,
          in1T->strides()[0] * elemSize, tWInput, dst_ptr);
          dst_ptr += 8 * elemSize;
      }

      // Set mask for last VReg in group.
      __asm__ __volatile__ (
        "mov.m.x m0, %[tail_mask], 0x0\n"
        :
        : [tail_mask] "r" (dstRowTailVRegMask)
      );

      embeddingBagsTailVectorized<elK>(minionCurrIndex,
        currSegmentStart, currSegmentEnd,
        tAInput, indices,
        in1T->strides()[0] * elemSize, tWInput, dst_ptr);

      minionCurrIndex += (currSegmentEnd - currSegmentStart);

      // Move from row tail to next row.
      minionCurrSegment++;
      minionCurrRowGroup = 0;

      getNextSegment(minionCurrSegment);

      dst_ptr = tOutput + minionCurrSegment * outT->strides()[0] * elemSize + minionCurrRowGroup * 64;
    }
  }
  
  outT->evict(DO_EVICTS);
}

template <ElemKind elK>
inline __attribute((always_inline))
void fwdLibEmbeddingBagInstBest(const int desired, LibTensor* outT, LibTensor *in1T, LibTensor* in2T, 
                                 LibTensor* in3T, LibTensor* in4T, bool hasEndOffset,
                                 uint64_t flags, const uint32_t minionOffset = 0,
                                 const uint32_t assignedMinions = 0) {

  switch(desired){
  case 1: inlining::fwdLibEmbeddingBagInst<elK>(outT, in1T, in2T, in3T, in4T, hasEndOffset, flags, minionOffset, assignedMinions); break;
  case 2: inlining::fwdLibEmbeddingBagInstVectorized<elK>(outT, in1T, in2T, in3T, in4T, hasEndOffset, flags, minionOffset, assignedMinions); break;
  default:
    {
#ifdef  SW_3755
      if (outT->getUntouchable())
        inlining::fwdLibEmbeddingBagInst<elK>(outT, in1T, in2T, in3T, in4T, hasEndOffset, flags, minionOffset, assignedMinions);
      else
#endif
        inlining::fwdLibEmbeddingBagInstVectorized<elK>(outT, in1T, in2T, in3T, in4T, hasEndOffset, flags, minionOffset, assignedMinions);
    }
    break;
  }
}

} // namespace inlining
} // namespace dnn_lib

#endif // _EMBEDDING_BAG_INST_H_
