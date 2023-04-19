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
#include <cmath>
#include <fenv.h>
#include <limits>
#include <string.h>
#include <type_traits>

#include "Float16.h"
#include "LibCommon.h"
#include "LibTensor.h"
#include "LibTypes.h"
#include "etsoc/isa/atomic.h"
#include "utils.h" // From include/internal path

// static bool enablePrinting;

/**
 * @brief packs 8 fp16s into a contiguous region of the f reg and and stores it to global mem
 * @param[in] FP_REG_ register name (will be stringified)
 * @param[in] DST_PTR_ dst ptr to store (shall be cacheline-aligned)
 */

#define PACK_AND_GLOBAL_STORE_8_FP16(FP_REG_, DST_PTR_)                                                                \
  do {                                                                                                                 \
    volatile uint64_t stReg0 = 0;                                                                                      \
    volatile uint64_t stReg1 = 0;                                                                                      \
    volatile uint64_t stReg2 = 0;                                                                                      \
    volatile uint64_t stReg3 = 0;                                                                                      \
    __asm__ __volatile__(/* pack the downconverted fp16 to 128 consecutive bits */                                     \
                         "fpackreph.pi " #FP_REG_ "," #FP_REG_ "\n" /* split 128 bits block in 4 32 bit pieces*/       \
                         "fmvz.x.ps %[stReg0], " #FP_REG_ ", 0\n"                                                      \
                         "fmvz.x.ps %[stReg1], " #FP_REG_ ", 1\n"                                                      \
                         "fmvz.x.ps %[stReg2], " #FP_REG_ ", 2\n"                                                      \
                         "fmvz.x.ps %[stReg3], " #FP_REG_ ", 3\n" /*shift 2nd and 4th word */                          \
                         "slli  %[stReg1], %[stReg1], 32\n"                                                            \
                         "slli  %[stReg3], %[stReg3], 32\n" /*or 1srt and 3rd */                                       \
                         "or %[stReg1], %[stReg1], %[stReg0]\n"                                                        \
                         "or %[stReg3], %[stReg3], %[stReg2]\n" /* global-store 128 bits in 2 ops. */                  \
                         "amoswapg.d x0, %[stReg1], (%[dst_ptr])\n"                                                    \
                         "addi        %[dst_ptr], %[dst_ptr], 8\n"                                                     \
                         "amoswapg.d x0, %[stReg3], (%[dst_ptr])\n"                                                    \
                         "addi        %[dst_ptr], %[dst_ptr], 8\n"                                                     \
                         : [ dst_ptr ] "+&r"(DST_PTR_)                                                                 \
                         : [ stReg0 ] "r"(stReg0), [ stReg1 ] "r"(stReg1), [ stReg2 ] "r"(stReg2),                     \
                           [ stReg3 ] "r"(stReg3)                                                                      \
                         : #FP_REG_);                                                                                  \
  } while (0)

/**
 * @brief packs 16 fp16s (8 in each f reg)  into a contiguous f reg and and stores it to global mem
 * @param[in] FP_REG_1_ register name containing the first fp16s (will be stringified)
 * @param[in] FP_REG_2_ register name containing the second fp16s (will be stringified)
 * @param[in] DST_PTR_ dst ptr to store (shall be cacheline-aligned)
 */

#define PACK_AND_GLOBAL_STORE_16_FP16(FP_REG_0_, FP_REG_1_, DST_PTR_)                                                  \
  do {                                                                                                                 \
    __asm__ __volatile__("mov.m.x  m0, zero, 0xf \n"                                                                   \
                         "fpackreph.pi " #FP_REG_0_ "," #FP_REG_0_ "\n" /* pack 8 fp16s in the lower half */           \
                         "mov.m.x  m0, zero, 0xf0 \n"                                                                  \
                         "fpackreph.pi " #FP_REG_0_ "," #FP_REG_1_ "\n" /* pack 8 fp16s in the higher half */          \
                         "mov.m.x  m0, zero, 0xfF \n"                   /* restore the mask */                         \
                         "fswg.ps " #FP_REG_0_ ", (%[dst_ptr])\n"       /* gloobal store the whole  256 bits reg */    \
                         "addi        %[dst_ptr], %[dst_ptr], 32\n"                                                    \
                         : [ dst_ptr ] "+&r"(DST_PTR_)                                                                 \
                         :                                                                                             \
                         : #FP_REG_0_, #FP_REG_1_);                                                                    \
  } while (0)

/**
 * @brief packs 8 fp16s into a contiguous region of the f reg and and stores it to mem
 * @param[in] FP_REG_ register name (will be stringified)
 * @param[in] DST_PTR_ dst ptr to store (shall be cacheline-aligned)
 */

#define PACK_AND_STORE_8_FP16(FP_REG_, DST_PTR_)                                                                       \
  do {                                                                                                                 \
    __asm__ __volatile__(/* pack the downconverted fp16 to 128 consecutive bits */                                     \
                         "fpackreph.pi " #FP_REG_ "," #FP_REG_ "\n" /* split 128 bits block in 4 32 bit pieces*/       \
                         "mov.m.x  m0, zero, 0x0f\n"                                                                   \
                         "fsw.ps " #FP_REG_ ", 0(%[dst_ptr])\n"                                                        \
                         "mov.m.x  m0, zero, 0xff\n"                                                                   \
                         : [ dst_ptr ] "+&r"(DST_PTR_)                                                                 \
                         :                                                                                             \
                         : #FP_REG_);                                                                                  \
  } while (0)

/**
 * @brief packs 16 fp16s (8 in each f reg)  into a contiguous f reg and and stores it to global mem
 * @param[in] FP_REG_1_ register name containing the first fp16s (will be stringified)
 * @param[in] FP_REG_2_ register name containing the second fp16s (will be stringified)
 * @param[in] DST_PTR_ dst ptr to store (shall be cacheline-aligned)
 */

#define PACK_AND_STORE_16_FP16(FP_REG_0_, FP_REG_1_, DST_PTR_)                                                         \
  do {                                                                                                                 \
    __asm__ __volatile__("mov.m.x  m0, zero, 0x0f\n"                                                                   \
                         "fpackreph.pi " #FP_REG_0_ "," #FP_REG_0_ "\n" /* pack 8 fp16s in the lower half */           \
                         "mov.m.x  m0, zero, 0xf0\n"                                                                   \
                         "fpackreph.pi " #FP_REG_0_ "," #FP_REG_1_ "\n" /* pack 8 fp16s in the higher half */          \
                         "mov.m.x  m0, zero, 0xff\n"                    /* restore the mask */                         \
                         "fsw.ps  " #FP_REG_0_ ", 0(%[dst_ptr])\n"      /* gloobal store the whole  256 bits reg */    \
                         "addi        %[dst_ptr], %[dst_ptr], 32\n"                                                    \
                         : [ dst_ptr ] "+&r"(DST_PTR_)                                                                 \
                         :                                                                                             \
                         : #FP_REG_0_, #FP_REG_1_);                                                                    \
  } while (0)

namespace dnn_lib {

namespace inlining {

inline __attribute((always_inline)) void pack_and_global_store_fp16x16(f32x8 src0, const f32x8 src1, void* dstPtr) {
  __asm__ __volatile__("mov.m.x  m0, zero, 0x0F \n"
                       "fpackreph.pi %[src0], %[src0]\n" /* pack 8 fp16s in the lower half */
                       "mov.m.x  m0, zero, 0xF0 \n"
                       "fpackreph.pi %[src0], %[src1]\n" /* pack 8 fp16s in the higher half */
                       "mov.m.x  m0, zero, 0xFF \n"      /* restore mask */
                       "fswg.ps %[src0], (%[dst_ptr])\n" /* global store the whole  256 bits reg */
                       : [ dst_ptr ] "+&r"(dstPtr), [ src0 ] "+&f"(src0)
                       : [ src1 ] "f"(src1)
                       :);
}

inline __attribute((always_inline)) void pack_and_store_fp16x16(f32x8 src0, const f32x8 src1, void* dstPtr) {
  __asm__ __volatile__("mov.m.x  m0, zero, 0x0F \n"
                       "fpackreph.pi %[src0], %[src0]\n" /* pack 8 fp16s in the lower half */
                       "mov.m.x  m0, zero, 0xF0 \n"
                       "fpackreph.pi %[src0], %[src1]\n" /* pack 8 fp16s in the higher half */
                       "mov.m.x  m0, zero, 0xFF \n"      /* restore mask */
                       "fsw.ps %[src0], 0(%[dst_ptr])\n" /* store the whole 256 bits reg */
                       : [ dst_ptr ] "+&r"(dstPtr), [ src0 ] "+&f"(src0)
                       : [ src1 ] "f"(src1)
                       :);
}

INLINE_ATTR uint32_t getVMask(int64_t n) {
  // VLEN is 8.
  // If n >= 31, return a mask with all ones (0xFFFFFFFF), otherwise, a mask with n 1's set.

  uint32_t gvl = n > 31 ? ~0u : (1 << n) - 1;
  gvl = n < 0 ? 0 : gvl; // if n is negative, mask is 0x0000000
  return gvl;
}

template <ElemKind elK, ElemKind indexElK>
inline __attribute((always_inline)) void
embeddingBagsTailVectorized(uintptr_t minionCurrIndex, uintptr_t currSegmentStart, uintptr_t currSegmentEnd,
                            uint8_t* tAInput, void* indicesPtr, uintptr_t dataRowPitch, uintptr_t dataRowGroupOffset,
                            uint8_t* tWInput, uint8_t* dst_ptr, bool destAlignedVregFP32, bool destAlignedVregFP16,
                            bool destGlobalReq) {
  constexpr bool float32Dst = (elK == FloatTy);
  constexpr bool float16Dst = (elK == Float16Ty);
  constexpr bool indices32B = (indexElK == Int32ITy);

  static_assert((elK == FloatTy) || (elK == Float16Ty));
  static_assert((indexElK == Int32ITy) || (indexElK == Int64ITy));

  constexpr uintptr_t elemSize = float32Dst ? 4 : 2;

  // Clear vector accumulator at the start.
  __asm__ __volatile__ (
    "fxor.pi f0, f0, f0\n"
    : 
    :
    : "f0"
  );

  // For all sparse input rows.
  for (uintptr_t j = currSegmentStart, currIndex = minionCurrIndex; j < currSegmentEnd; j++, currIndex++) {
    int64_t rowOffset;
    if (indices32B) {
      int32_t* indicesPtr32B = (int32_t*)indicesPtr;
      rowOffset = indicesPtr32B[currIndex];
    } else {
      int64_t* indicesPtr64B = (int64_t*)indicesPtr;
      rowOffset = indicesPtr64B[currIndex];
    }

    uint8_t* data_ptr = tAInput + rowOffset * dataRowPitch + dataRowGroupOffset;
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
      __asm__ __volatile__("flw.ps     f10, 0(%[data_ptr])\n" : : [ data_ptr ] "r"(data_ptr) : "f10");
      data_ptr += 32;
    } else {
      __asm__ __volatile__("fgh.ps      f10, f20(%[data_ptr])\n"
                           "fcvt.ps.f16 f10, f10\n"
                           :
                           : [ data_ptr ] "r"(data_ptr)
                           : "f10");
    }

    __asm__ __volatile__ (
      "fmadd.ps f0, f21, f10, f0\n"
      :
      :
      : "f0"
    );
  }

  if (float32Dst) {
    // Store sequence for FP32 results
    // Global stores are required as different minions will write to same cache line
    if (destGlobalReq) {
      if (destAlignedVregFP32) {
        // If results are aligned to full vector of 32bits, can use regular store global
        __asm__ __volatile__("fswg.ps f0, (%[dst_ptr])\n" : : [ dst_ptr ] "r"(dst_ptr) : "f0");
      } else {
        // Need to use global scatter, as results are not aligned to full vector
        __asm__ __volatile__("fscwg.ps f0, f20(%[dst_ptr])\n" : : [ dst_ptr ] "r"(dst_ptr) : "f0", "f20");
      }
    } else {
      // Can use regular store as minions won't collide in same cacheline
      __asm__ __volatile__("fsw.ps f0, 0(%[dst_ptr])\n" : : [ dst_ptr ] "r"(dst_ptr) : "f0");
    }
  } else {
    // Store sequence for FP16 results
    __asm__ __volatile__ (
      "fcvt.f16.ps f0, f0\n"
      :
      :
      : "f0"
    );
    // Global stores are required as different minions will write to same cache line
    if (destGlobalReq) {
      if (destAlignedVregFP16) {
        // If results are aligned to full vector of 16bits, can use regular store global with pack
        PACK_AND_GLOBAL_STORE_8_FP16(f0, dst_ptr);
      } else {
        // Need ot use global scatter, as resutls are not aligned to full vector
        __asm__ __volatile__("fschg.ps f0, f20(%[dst_ptr])\n" : : [ dst_ptr ] "r"(dst_ptr) : "f0", "f20");
      }
    } else {
      PACK_AND_STORE_8_FP16(f0, dst_ptr);
    }
  }
}

template <ElemKind outElK, ElemKind dataElK, ElemKind indexElK>
inline __attribute((always_inline)) typename std::enable_if_t<(dataElK == Int8QTy && outElK == FloatTy), void>
embeddingBagsTailVectorized(dim_t minionCurrIndex, dim_t currSegmentStart, dim_t currSegmentEnd, int8_t* tAInput,
                            const float scale, const int32_t offset, int64_t* indices, dim_t dataRowPitch,
                            dim_t dataRowGroupOffset, float* tWInput, float* dst_ptr, bool alignedStores,
                            const f32x8 int8Offsets) {

  constexpr int32_t simdWidth = 8;
  using outType = typename elemKind2elemTy<outElK>::type;
  using dataType = typename elemKind2elemTy<dataElK>::type;

  // Clear vector accumulator at the start.
  f32x8 accum;
  __asm__ __volatile__("fxor.pi %[accum], %[accum], %[accum]\n" : [ accum ] "=&f"(accum) : :);

  // For all sparse input rows.
  for (dim_t j = currSegmentStart, currIndex = minionCurrIndex; j < currSegmentEnd; j++, currIndex++) {
    dataType* data_ptr = tAInput + indices[currIndex] * dataRowPitch + dataRowGroupOffset;
    outType* weight_ptr = tWInput + currIndex;

    f32x8 weightValues;
    __asm__ __volatile__("fbc.ps  %[weightValues], 0x0(%[weight_ptr])\n"
                         : [ weightValues ] "=&f"(weightValues)
                         : [ weight_ptr ] "r"(weight_ptr)
                         :);

    f32x8 dataValues;
    f32x8 offsetVector;
    f32x8 scaleVector;

    // Dequantize
    __asm__ __volatile__(
      // Load 1x8 Int8QTy elements of data into 1x8 Int32Qty
      "fgb.ps %[dataValues], %[int8Offsets](%[data_ptr])\n"
      // Broadcast scale and offset
      "fbcx.ps %[offsetVector], %[offset]\n"
      "fbcx.ps %[scaleVector], %[scale]\n"
      // dataValue = scale * ( dataValue - offset )
      // 1. Sub offset
      "fsub.pi %[dataValues], %[dataValues], %[offsetVector]\n"
      // 2. Convert values from Int32 to FP32
      "fcvt.ps.pw %[dataValues], %[dataValues]\n"
      // 3. Mul by scale
      "fmul.ps %[dataValues], %[dataValues], %[scaleVector]\n"
      : [ offsetVector ] "=&f"(offsetVector), [ scaleVector ] "=&f"(scaleVector), [ dataValues ] "=&f"(dataValues)
      : [ int8Offsets ] "f"(int8Offsets), [ data_ptr ] "r"(data_ptr), [ offset ] "r"(offset), [ scale ] "r"(scale)
      :);
    data_ptr += simdWidth;

    __asm__ __volatile__(
      // multipy datavalues by weight and accumulate to prev value
      "fmadd.ps %[accum], %[dataValues], %[weightValues], %[accum]\n"
      : [ accum ] "+&f"(accum)
      : [ dataValues ] "f"(dataValues), [ weightValues ] "f"(weightValues)
      :);
  }
  // Store accumulated results.
  if (alignedStores) {
    __asm__ __volatile__("fswg.ps %[accum], (%[dst_ptr])\n" : : [ dst_ptr ] "r"(dst_ptr), [ accum ] "f"(accum) :);
  } else {
    f32x8 fp32Offsets;
    __asm__ __volatile__("fslli.pi %[fp32Offsets], %[int8Offsets], 2\n"
                         "fscwg.ps %[accum], %[int8Offsets](%[dst_ptr])\n"
                         : [ fp32Offsets ] "=&f"(fp32Offsets)
                         : [ dst_ptr ] "r"(dst_ptr), [ accum ] "f"(accum), [ int8Offsets ] "f"(int8Offsets)
                         :);
  }
}

template <ElemKind indexElK>
INLINE_ATTR bool getNextSegment(const dim_t segment, const dim_t segments, const dim_t numIndices, LibTensor* offsets,
                                const bool hasEndOffset, dim_t& currSegmentStart, dim_t& currSegmentEnd) {
  using indicesType = typename elemKind2elemTy<indexElK>::type;
  auto offH = offsets->getHandle<indicesType>();
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

  if (start >= end) {
    emptySegment = true;
  } else {
    emptySegment = false;
  }
  currSegmentStart = start;
  currSegmentEnd = end;
  return emptySegment;
}

INLINE_ATTR std::pair<dim_t, dim_t> getWorkDistribution(size_t minionId, size_t activeMinions, dim_t totalWorkUnits) {
  // If workBalancing is enabled the work will be partitioned based on the amount of indices to reduce per segment.
  // (Only useful if the indices per segment vary significantly)
  // else, the work partitioning will distribute evenly the segments across minions.
  constexpr bool workBalancing = false;
  dim_t minionWorkUnits = 0;
  dim_t minionFirstWorkUnit = 0;

  if constexpr (workBalancing) {
    // TODO
  } else {
    if ((totalWorkUnits % activeMinions) == 0) {
      // Num of cache lines is multiple of activeMinions

      // Compute the number of cache lines this minion will work on
      minionWorkUnits = totalWorkUnits / activeMinions;
      // Compute the index into the first work unit (cache line).
      minionFirstWorkUnit = minionId * minionWorkUnits;
    } else {
      // Num of cache lines is NOT multiple of activeMinions
      minionWorkUnits = totalWorkUnits / activeMinions;
      const dim_t remainingWorkUnits = totalWorkUnits % activeMinions;

      if (minionId < remainingWorkUnits) {
        // Some minions will do more work
        minionWorkUnits++;
        // Compute the index into the first work unit.
        minionFirstWorkUnit = minionId * minionWorkUnits;
      } else {
        // Compute the index into the first work unit.
        minionFirstWorkUnit =
          remainingWorkUnits * (minionWorkUnits + 1) + (minionId - remainingWorkUnits) * minionWorkUnits;
      }
    }
  }
  return {minionWorkUnits, minionFirstWorkUnit};
}

template <ElemKind outElK, ElemKind in1ElK, ElemKind indexElK>
INLINE_ATTR typename std::enable_if_t<(in1ElK == FloatTy || in1ElK == Float16Ty), void>
fwdLibEmbeddingBagInstVectorized(LibTensor* outT, LibTensor* data, LibTensor* weights, LibTensor* indices,
                                 LibTensor* offsets, bool hasEndOffset, uint64_t flags, const uint32_t minionOffset = 0,
                                 const uint32_t assignedMinions = 0) {
  et_assert(data->getElementType() == outT->getElementType());
  et_assert((data->getElementType() == FloatTy) || (data->getElementType() == Float16Ty));
  et_assert(indices->getElementType() == offsets->getElementType());
  et_assert((indices->getElementType() == Int64ITy) || (indices->getElementType() == Int32ITy));

  // in This instantiation, in and out forcibly match.
  static_assert(in1ElK == outElK);
  constexpr bool float32Dst = (outElK == FloatTy);
  constexpr bool float16Dst = (outElK == Float16Ty);

  static_assert(outElK == FloatTy || outElK == Float16Ty);
  // Get size of the output element.
  const uintptr_t elemSize = float32Dst ? 4 : 2;
  // Get first ID for the first minion assigned to this operation.
  uint64_t minionId = get_minion_id();

  // If Minion is outside the group assigned to this Node get out.
  if (minionId < minionOffset) {
    return;
  }

  // Rebase minion ID.
  minionId -= minionOffset;

  // Get number of Minions assigned to this Node.
  uint64_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;

  // If Minion is outside the group assigned to this Node get out.
  if (minionId >= activeMinions) {
    return;
  }

  auto tOutput = outT->getRawDataPointer<uint8_t>();
  auto tAInput = data->getRawDataPointer<uint8_t>();
  auto tWInput = weights->getRawDataPointer<uint8_t>();
  using indicesType = typename elemKind2elemTy<indexElK>::type;
  auto indicesPtr = indices->getRawDataPointer<indicesType>();
  auto offH = offsets->getHandle<indicesType>();

  const dim_t segments = hasEndOffset ? (offsets->dims()[0] - 1) : offsets->dims()[0];
  const dim_t numIndices = indices->dims()[0];

  // Compute the number of elements per data row (first tensor dimension).
  const uintptr_t dstRowElemSize = outT->dims()[1];  
  
  // Assign work to Minions :
  //
  // NOTE: NOT IMPLEMENTED!!!
  // If the row is smaller than a cache line then multiple rows need to
  // be assigned to a single Minion.

  // Compute the number of 8-element vectors (VRegs) per output cache line
  // (a Minion must be assigned a full cache line to avoid coherence issues
  // when writing).
  // This correspond with a group (one or more VRegs).
  const uintptr_t dstVRegElems = 8; // SIMD/Vector length  in elements
  const uintptr_t dstGroupElems = CACHE_LINE_BYTES /  elemSize;
  const uintptr_t dstGroupVRegs = CACHE_LINE_BYTES / (elemSize * 8);

  // Compute the number of groups per output row (rounded up).
  const uintptr_t dstRowGroups = ((dstRowElemSize - 1) / dstGroupElems) + 1;

  // To prevent issues with two minions writing on the same cacheline when
  // the destination register is not cacheline aligned, flag it here
  bool singleMinionPerRow = (((uint64_t)tOutput % CACHE_LINE_BYTES) != 0);

  // Computes if there's a tail
  bool dstRowHasTail = ((dstRowElemSize % dstGroupElems) != 0);

  // Compute the number of 8-element vectors in the tail of the row
  // (number of VRegs not multiple of Group VRegs).
  uintptr_t dstRowTailVRegs = (((dstRowElemSize - 1) / dstVRegElems) + 1) % dstGroupVRegs;
  if (dstRowTailVRegs == 0) {
    dstRowTailVRegs += dstGroupVRegs;
  }

  // Compute the element mask for the last VReg in the row.
  int dstRowTailVRegMask = (1 << (dstRowElemSize % dstVRegElems)) - 1;
  // If 0, it means all the lanes need to be enabled in last pass
  if (dstRowTailVRegMask == 0) {
    dstRowTailVRegMask = (1 << dstVRegElems) - 1;
  }

  uintptr_t totalWorkUnits = dstRowGroups * outT->dims()[0];

  //  Distribute the tail of groups.
  uintptr_t minionWorkUnits = 0;
  uintptr_t minionFirstWorkUnit = 0;

  // If a row is done by a single minion, we need to distribute work in a per row granularity
  if (singleMinionPerRow) {
    // Even distribution across minions
    if ((outT->dims()[0] % activeMinions) == 0) {
      minionWorkUnits = outT->dims()[0] / activeMinions;
      // Converts from rows to work units
      minionWorkUnits *= dstRowGroups;
      minionFirstWorkUnit = minionId * minionWorkUnits;
    } else {
      // Uneven distribution across minions
      minionWorkUnits = outT->dims()[0] / activeMinions;
      const uintptr_t remainingWorkUnits = outT->dims()[0] % activeMinions;
      if (minionId < remainingWorkUnits) {
        minionWorkUnits++;
        // Converts from rows to work units
        minionWorkUnits *= dstRowGroups;
        // Compute the index into the first work unit.
        minionFirstWorkUnit = minionId * minionWorkUnits;
      } else {
        // Converts from rows to work units
        minionWorkUnits *= dstRowGroups;
        // Compute the index into the first work unit.
        minionFirstWorkUnit =
          remainingWorkUnits * (minionWorkUnits + dstRowGroups) + (minionId - remainingWorkUnits) * minionWorkUnits;
      }
    }
  } else {
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
        minionFirstWorkUnit =
          remainingWorkUnits * (minionWorkUnits + 1) + (minionId - remainingWorkUnits) * minionWorkUnits;
      }
    }
  }

  // No work for this Minion.
  if (minionWorkUnits == 0) {
    return;
  }

  // Computes if the destination is correctly aligned to vReg and contiguous stores
  // can be used isntead of scatters.
  // Need both the dest starting address being VReg aligned as well as the pitch for
  // the smallest dimension
  // The padding must be touchable or the number of elements multiple of Vreg size
  const auto outputStrideZeroBytes = outT->strides()[0] * elemSize;
  bool destAlignedVregFP32 = (((uint64_t)tOutput % VREG_BYTES) == 0) && ((outputStrideZeroBytes % VREG_BYTES) == 0) &&
                             (!outT->getUntouchable() || (((uint64_t)outT->dims()[1] % dstVRegElems) == 0));
  bool destAlignedVregFP16 = (((uint64_t)tOutput % (VREG_BYTES / 2)) == 0) &&
                             ((outputStrideZeroBytes % (VREG_BYTES / 2)) == 0) &&
                             (!outT->getUntouchable() || (((uint64_t)outT->dims()[1] % dstVRegElems) == 0));
  // We require to use global stores to destination if more than two minions might write in the same cacheline
  bool destGlobalReq = (((outT->strides()[0] * elemSize) % CACHE_LINE_BYTES) != 0);

  // Compute the first output row (segment) assigned to the Minion.
  uintptr_t minionFirstSegment = minionFirstWorkUnit / dstRowGroups;

  // Compute current group in row assigned to the Minion.
  uintptr_t minionFirstRowGroup = minionFirstWorkUnit % dstRowGroups;

  // Get the first index assigned to the Minion.
  uintptr_t minionFirstIndex = 0;
  uintptr_t currSegmentStart = 0;
  uintptr_t currSegmentEnd   = 0;

  if (offH.raw(0) == 0) {
    // fast path, offests start at 0, we can have a direct access to minionFirstIndex
    minionFirstIndex = offH.raw(minionFirstSegment);
    (void)getNextSegment<indexElK>(minionFirstSegment, segments, numIndices, offsets, hasEndOffset, currSegmentStart,
                                   currSegmentEnd);
  } else {
    // slow path, offests do not start at 0, offets need traveral to get to minionFirstIndex
    bool minionEmptySegment = false;
    for (uintptr_t s = 0; s <= minionFirstSegment; s++) {
      if (not minionEmptySegment) {
        minionFirstIndex += (currSegmentEnd - currSegmentStart);
      }
      minionEmptySegment =
        getNextSegment<indexElK>(s, segments, numIndices, offsets, hasEndOffset, currSegmentStart, currSegmentEnd);
    }
  }

  // Initialize indices.
  uintptr_t minionCurrIndex    = minionFirstIndex;
  uintptr_t minionCurrSegment  = minionFirstSegment;
  uintptr_t minionCurrRowGroup = minionFirstRowGroup;

  // Initilize output pointer.
  auto dst_ptr = tOutput + minionCurrSegment * outputStrideZeroBytes + minionCurrRowGroup * dstGroupElems * elemSize;

  // Prepare gather indices
  int32_t gather_offsets16[] = { 0, 2, 4, 6, 8, 10, 12, 14 };

  __asm__ __volatile__("mov.m.x m0, zero, 0xff\n"
                       "flw.ps  f20, %[gather_offsets16]\n"
                       :
                       : [ gather_offsets16 ] "m"(*(const int32_t(*)[8])gather_offsets16)
                       : "f20");

  // f20 is doubled in case of FP32 and needing not aligned stores
  if (float32Dst) {
    __asm__ __volatile__ (
      "fslli.pi f20, f20, 1\n"
      :
      :
      : "f20"
    );
  }

  // For all minion assigned work units
  for (uintptr_t i = 0; i < minionWorkUnits; i++) {

    // Detect row tail
    bool dstGroupNotInRowTail = !dstRowHasTail || (minionCurrRowGroup != (dstRowGroups - 1));

    if (dstGroupNotInRowTail) {
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
          :
          :
          : "f2", "f3"
       );
      }
      // For all sparse input rows.
      for (uintptr_t j = currSegmentStart, currIndex = minionCurrIndex;
           j < currSegmentEnd; j++, currIndex++) {
        uint8_t* data_ptr =
          tAInput +
          (((int64_t)indicesPtr[currIndex]) * data->strides()[0] + minionCurrRowGroup * dstGroupElems) * elemSize;
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

        if (j < currSegmentEnd - 1) {
          uint8_t* data_ptr_next =
            tAInput +
            (((int64_t)indicesPtr[currIndex + 1]) * data->strides()[0] + minionCurrRowGroup * dstGroupElems) * elemSize;
          // Prefetch next index of current segment
          __asm__ __volatile__("ld          x0, (%[data_ptr_next])\n" : : [ data_ptr_next ] "r"(data_ptr_next) :);
        }

        if (float16Dst) {
          __asm__ __volatile__("fgh.ps      f10, f20(%[data_ptr])\n"
                               "addi        %[data_ptr], %[data_ptr], 16\n"
                               "fgh.ps      f11, f20(%[data_ptr])\n"
                               "addi        %[data_ptr], %[data_ptr], 16\n"
                               "fgh.ps      f12, f20(%[data_ptr])\n"
                               "addi        %[data_ptr], %[data_ptr], 16\n"
                               "fgh.ps      f13, f20(%[data_ptr])\n"
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
                               : [ data_ptr ] "r"(data_ptr)
                               : "f0", "f1", "f2", "f3", "f10", "f11", "f12", "f13");
        } else {    // Float32
          __asm__ __volatile__("flw.ps   f10,  0(%[data_ptr])\n"
                               "flw.ps   f11, 32(%[data_ptr])\n"
                               "fmadd.ps f0, f21, f10, f0\n"
                               "fmadd.ps f1, f21, f11, f1\n"
                               :
                               : [ data_ptr ] "r"(data_ptr)
                               : "f0", "f1", "f10", "f11");
        }
      }

      if (float32Dst) {
        if (destGlobalReq) {
          if (destAlignedVregFP32) {
            // Store accumulated results.
            __asm__ __volatile__("fswg.ps f0,  (%[dst_ptr])\n"
                                 "addi    %[dst_ptr], %[dst_ptr], 32\n"
                                 "fswg.ps f1,  (%[dst_ptr])\n"
                                 "addi    %[dst_ptr], %[dst_ptr], 32\n"
                                 : [ dst_ptr ] "+&r"(dst_ptr)
                                 :
                                 : "f0", "f1");
          } else {
            __asm__ __volatile__("fscwg.ps    f0, f20(%[dst_ptr])\n"
                                 "addi        %[dst_ptr], %[dst_ptr], 32\n"
                                 "fscwg.ps    f1, f20(%[dst_ptr])\n"
                                 "addi        %[dst_ptr], %[dst_ptr], 32\n"
                                 : [ dst_ptr ] "+&r"(dst_ptr)
                                 :
                                 : "f0", "f1", "f20");
          }
        } else {
          // Store accumulated results.
          __asm__ __volatile__("fsw.ps f0, 0(%[dst_ptr])\n"
                               "addi    %[dst_ptr], %[dst_ptr], 32\n"
                               "fsw.ps f1, 0(%[dst_ptr])\n"
                               "addi    %[dst_ptr], %[dst_ptr], 32\n"
                               : [ dst_ptr ] "+&r"(dst_ptr)
                               :
                               : "f0", "f1");
        }
      } else { // Float16
        // Convert and store accumulated results.
        __asm__ __volatile__("fcvt.f16.ps f0, f0\n"
                             "fcvt.f16.ps f1, f1\n"
                             "fcvt.f16.ps f2, f2\n"
                             "fcvt.f16.ps f3, f3\n"
                             :
                             :
                             : "f0", "f1", "f2", "f3");

        if (destGlobalReq) {
          if (destAlignedVregFP32) {
            PACK_AND_GLOBAL_STORE_16_FP16(f0, f1, dst_ptr);
            PACK_AND_GLOBAL_STORE_16_FP16(f2, f3, dst_ptr);
          } else {
            __asm__ __volatile__("fschg.ps     f0, f20(%[dst_ptr])\n"
                                 "addi        %[dst_ptr], %[dst_ptr], 16\n"
                                 "fschg.ps     f1, f20(%[dst_ptr])\n"
                                 "addi        %[dst_ptr], %[dst_ptr], 16\n"
                                 "fschg.ps     f2, f20(%[dst_ptr])\n"
                                 "addi        %[dst_ptr], %[dst_ptr], 16\n"
                                 "fschg.ps     f3, f20(%[dst_ptr])\n"
                                 "addi        %[dst_ptr], %[dst_ptr], 16\n"
                                 : [ dst_ptr ] "+&r"(dst_ptr)
                                 :
                                 : "f0", "f1", "f2", "f3", "f20");
          }
        } else {
          PACK_AND_STORE_16_FP16(f0, f1, dst_ptr);
          PACK_AND_STORE_16_FP16(f2, f3, dst_ptr);
        }
      }

      if (minionCurrRowGroup != (dstRowGroups - 1)) {
        minionCurrRowGroup++;
      } else {
        // Move from row tail to next row.
        minionCurrSegment++;
        minionCurrRowGroup = 0;

        if (currSegmentEnd > currSegmentStart) {
          minionCurrIndex += (currSegmentEnd - currSegmentStart);
        }

        getNextSegment<indexElK>(minionCurrSegment, segments, numIndices, offsets, hasEndOffset, currSegmentStart,
                                 currSegmentEnd);
        dst_ptr = tOutput + minionCurrSegment * outputStrideZeroBytes + minionCurrRowGroup * dstGroupElems * elemSize;
      }
    } else {
      // Initialize mask to clear upper bytes from input load
      __asm__ __volatile__ (
        "mov.m.x m0, zero, 0xff\n"
      );

      for (uintptr_t k = 0; k < (dstRowTailVRegs - 1); k++) {
        embeddingBagsTailVectorized<outElK, indexElK>(
          minionCurrIndex, currSegmentStart, currSegmentEnd, tAInput, (void*)indicesPtr, data->strides()[0] * elemSize,
          (minionCurrRowGroup * dstGroupElems + k * dstVRegElems) * elemSize, tWInput, dst_ptr, destAlignedVregFP32,
          destAlignedVregFP16, destGlobalReq);
        dst_ptr += dstVRegElems * elemSize;
      }

      // Set mask for last VReg in group.
      __asm__ __volatile__ (
        "mov.m.x m0, %[tail_mask], 0x0\n"
        :
        : [tail_mask] "r" (dstRowTailVRegMask)
      );

      embeddingBagsTailVectorized<outElK, indexElK>(
        minionCurrIndex, currSegmentStart, currSegmentEnd, tAInput, (void*)indicesPtr, data->strides()[0] * elemSize,
        (minionCurrRowGroup * dstGroupElems + (dstRowTailVRegs - 1) * dstVRegElems) * elemSize, tWInput, dst_ptr,
        destAlignedVregFP32, destAlignedVregFP16, destGlobalReq);

      minionCurrIndex += (currSegmentEnd - currSegmentStart);

      // Move from row tail to next row.
      minionCurrSegment++;
      minionCurrRowGroup = 0;

      getNextSegment<indexElK>(minionCurrSegment, segments, numIndices, offsets, hasEndOffset, currSegmentStart,
                               currSegmentEnd);

      dst_ptr = tOutput + minionCurrSegment * outputStrideZeroBytes + minionCurrRowGroup * dstGroupElems * elemSize;
    }
  }
}

template <ElemKind outElK, ElemKind dataElK, ElemKind indexElK>
INLINE_ATTR typename std::enable_if_t<(dataElK == Int8QTy && (outElK == FloatTy || outElK == Float16Ty)), void>
fwdLibEmbeddingBagInstVectorized(LibTensor* out, LibTensor* data, LibTensor* weights, LibTensor* indices,
                                 LibTensor* offsets, bool hasEndOffset, uint64_t flags, const uint32_t minionOffset = 0,
                                 const uint32_t assignedMinions = 0) {

  et_assert(weights->getElementType() == out->getElementType());
  et_assert(indices->getElementType() == offsets->getElementType());
  et_assert((indices->getElementType() == Int64ITy) || (indices->getElementType() == Int32ITy));
  et_assert(reinterpret_cast<uint64_t>(out->getRawDataPointer<uint64_t>()) % 64 == 0);

  using outType = typename elemKind2elemTy<outElK>::type;
  using inType = typename elemKind2elemTy<dataElK>::type;
  using indicesType = typename elemKind2elemTy<indexElK>::type;

  // If Minion is outside the group assigned to this Node get out.
  size_t minionId = get_minion_id();
  if (minionId < minionOffset) {
    return;
  }

  // Rebase minion ID.
  minionId -= minionOffset;

  // Get number of Minions assigned to this Node.
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;

  // If Minion is outside the group assigned to this Node get out.
  if (minionId >= activeMinions) {
    return;
  }

  auto offH = offsets->getHandle<indicesType>();
  auto tWInput = weights->getRawDataPointer<outType>();

  // Work partitioning is based on output tensor 'segments'
  auto numSegments = out->dims()[0];
  // If the EB input data has 1 dimension x = n, is equivalent to a EB 2D where (x,y) = n,1
  dim_t segmentLen = out->ndims() > 1 ? out->dims()[1] : static_cast<dim_t>(1UL);

  dim_t numSegmentsPerMinion = std::max(((numSegments + activeMinions - 1) / activeMinions), 1UL);
  // This minion gets assigned to compute from startSegment to endSegment
  dim_t startSegment = numSegmentsPerMinion * minionId;
  if (startSegment >= numSegments) {
    return;
  }
  auto endSegment = std::min(startSegment + numSegmentsPerMinion, numSegments);

  // Handles
  auto IH = indices->getHandle<indicesType>();
  auto DH = data->getHandle<inType>();
  auto OH = out->getHandle<outType>();

  // Get the sizes to iterate
  // Pre-compute the indices used in gather/scatter operations
  // load gather for quantized Int8QTy data values
  // store scatter for FP16/FP32 accumulated results
  int32_t gather_offsets8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  f32x8 int8Offsets, fp16Offsets, fp32Offsets;
  const int32_t offset = data->getOffset();
  const float scale = data->getScale();
  __asm__ __volatile__("flw.ps  %[int8Offsets], %[gather_offsets8]\n"
                       "fslli.pi  %[fp16Offsets], %[int8Offsets], 1\n"
                       "fslli.pi  %[fp32Offsets], %[fp16Offsets], 1\n"
                       : [ int8Offsets ] "=&f"(int8Offsets), [ fp16Offsets ] "=&f"(fp16Offsets),
                         [ fp32Offsets ] "=&f"(fp32Offsets)
                       : [ gather_offsets8 ] "m"(*(const int32_t(*)[8])gather_offsets8)
                       :);

  // Broadcast scale and offset used in dequantization
  f32x8 offsetVector;
  f32x8 scaleVector;
  __asm__ __volatile__("fbcx.ps %[offsetVector], %[offset]\n"
                       "fbcx.ps %[scaleVector], %[scale]\n"
                       : [ offsetVector ] "=&f"(offsetVector), [ scaleVector ] "=&f"(scaleVector)
                       : [ offset ] "r"(offset), [ scale ] "r"(scale)
                       :);

  // Main loop
  for (dim_t s = startSegment; s < endSegment; s++) {
    f32x8 dataValues0;
    f32x8 accum0;

    // Get start and end offset of this segment
    int64_t startOffset = offH.raw(s);
    int64_t endOffset;
    if (s == (numSegments - 1UL)) {
      endOffset = hasEndOffset ? static_cast<int64_t>(offH.raw(s + 1)) : static_cast<int64_t>(indices->dims()[0]);
    } else {
      endOffset = static_cast<int64_t>(offH.raw(s + 1));
    }

    // Get the destination pointer to write results
    std::array<dim_t, 2> segmentCoord = {s, 0};
    auto dstPtr = &OH.at(segmentCoord);

    // Enable all bits in mask
    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : :);

    if constexpr (outElK == Float16Ty) {
      // 16 x FP16 elements on each iteration
      constexpr int64_t vlen = 16;
      f32x8 dataValues1;
      f32x8 accum1;

      int64_t i = 0;
      for (; i < (static_cast<int64_t>(segmentLen) - (vlen - 1LL)); i += vlen) {
        // Clear vector registers that will be used for accumulation
        __asm__ __volatile__("fxor.pi %[accum0], %[accum0], %[accum0]\n"
                             "fxor.pi %[accum1], %[accum1], %[accum1]\n"
                             : [ accum0 ] "=&f"(accum0), [ accum1 ] "=&f"(accum1)
                             :
                             :);

        // For every row in the segment
        for (int64_t j = startOffset; j < endOffset; j++) {
          // Load and broadcast weight and convert to FP32
          outType* weightPtr = tWInput + j;
          f32x8 weightValues;
          __asm__ __volatile__("fbc.ps  %[weightValues], 0x0(%[weightPtr])\n"
                               "fcvt.ps.f16 %[weightValues], %[weightValues]\n"
                               : [ weightValues ] "=&f"(weightValues)
                               : [ weightPtr ] "r"(weightPtr)
                               :);

          // Get the data source pointer
          auto rowIndex = static_cast<dim_t>(IH.raw(j));
          std::array<dim_t, 2> rowCoord = {rowIndex, static_cast<dim_t>(i)};
          auto srcPtr = reinterpret_cast<char*>(&DH.at(rowCoord));

          // Load 16 x Int8QTy data elements (srcPtr) and dequantize to FP32
          __asm__ __volatile__(
            "fgb.ps %[dataValues0],   %[int8Offsets](%[srcPtr])\n"
            "addi %[srcPtr], %[srcPtr], 8\n"
            "fgb.ps %[dataValues1],   %[int8Offsets](%[srcPtr])\n"
            "addi %[srcPtr], %[srcPtr], 8\n"
            // Dequantize: scale * (float)( (¡) input - (int32_t)offset )
            "fsub.pi %[dataValues0], %[dataValues0], %[offsetVector]\n"
            "fsub.pi %[dataValues1], %[dataValues1], %[offsetVector]\n"
            // convert to fp32
            "fcvt.ps.pw %[dataValues0], %[dataValues0]\n"
            "fcvt.ps.pw %[dataValues1], %[dataValues1]\n"
            // Mul by scale
            "fmul.ps %[dataValues0], %[dataValues0], %[scaleVector]\n"
            "fmul.ps %[dataValues1], %[dataValues1], %[scaleVector]\n"
            : [ dataValues0 ] "=&f"(dataValues0), [ dataValues1 ] "=&f"(dataValues1), [ srcPtr ] "+&r"(srcPtr)
            : [ int8Offsets ] "f"(int8Offsets), [ offsetVector ] "f"(offsetVector), [ scaleVector ] "f"(scaleVector)
            :);

          // Mutiply by weight and Accumulate with previous value
          __asm__ __volatile__("fmadd.ps %[accum0], %[dataValues0], %[weightValues], %[accum0]\n"
                               "fmadd.ps %[accum1], %[dataValues1], %[weightValues], %[accum1]\n"
                               : [ accum0 ] "+&f"(accum0), [ accum1 ] "+&f"(accum1)
                               : [ dataValues0 ] "f"(dataValues0), [ dataValues1 ] "f"(dataValues1),
                                 [ weightValues ] "f"(weightValues)
                               :);
        }
        // Convert to FP16 before storing
        __asm__ __volatile__("fcvt.f16.ps %[accum0], %[accum0]\n"
                             "fcvt.f16.ps %[accum1], %[accum1]\n"
                             : [ accum0 ] "+&f"(accum0), [ accum1 ] "+&f"(accum1)
                             :
                             :);
        pack_and_global_store_fp16x16(accum0, accum1, dstPtr);
        dstPtr += vlen;
      }

      if (i < static_cast<int64_t>(segmentLen)) { // This is the epilogue
        uint32_t mask0 = getVMask(static_cast<int64_t>(segmentLen) - i);
        uint32_t mask1 = mask0 & 0xFF00u; // Get bits [8:15] from mask0 to create mask1

        // reset mask before clearing accumulation registers
        __asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : :);
        // clear accumulation registers
        __asm__ __volatile__("fxor.pi %[accum0], %[accum0], %[accum0]\n"
                             "fxor.pi %[accum1], %[accum1], %[accum1]\n"
                             : [ accum0 ] "=&f"(accum0), [ accum1 ] "=&f"(accum1)
                             :
                             :);

        // For every row in the segment
        for (int64_t j = startOffset; j < endOffset; j++) {

          // reset mask before loading weights
          __asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : :);

          // Load and broadcast weight and convert to FP32
          outType* weightPtr = tWInput + j;
          f32x8 weightValues;
          __asm__ __volatile__("fbc.ps  %[weightValues], 0x0(%[weightPtr])\n"
                               "fcvt.ps.f16 %[weightValues], %[weightValues]\n"
                               : [ weightValues ] "=&f"(weightValues)
                               : [ weightPtr ] "r"(weightPtr)
                               :);

          // Get the data source pointer
          auto rowIndex = static_cast<dim_t>(IH.raw(j));
          std::array<dim_t, 2> rowCoord = {rowIndex, static_cast<dim_t>(i)};
          auto srcPtr = &DH.at(rowCoord);

          // Load 16 x Int8QTy data elements (srcPtr) and dequantize to FP32, accumulate in FP32.
          __asm__ __volatile__( // set mask0
            "mov.m.x m0, %[mask0], 0x0\n"
            "fgb.ps %[dataValues0],   %[int8Offsets](%[srcPtr])\n"
            "addi %[srcPtr], %[srcPtr], 8\n"
            "fsub.pi %[dataValues0], %[dataValues0], %[offsetVector]\n"
            "fcvt.ps.pw %[dataValues0], %[dataValues0]\n"
            "fmul.ps %[dataValues0], %[dataValues0], %[scaleVector]\n"
            "fmadd.ps %[accum0], %[dataValues0], %[weightValues], %[accum0]\n"
            // set mask1
            "mov.m.x m0, %[mask1], 0x0\n"
            "fgb.ps %[dataValues1],   %[int8Offsets](%[srcPtr])\n"
            "addi %[srcPtr], %[srcPtr], 8\n"
            "fsub.pi %[dataValues1], %[dataValues1], %[offsetVector]\n"
            "fcvt.ps.pw %[dataValues1], %[dataValues1]\n"
            "fmul.ps %[dataValues1], %[dataValues1], %[scaleVector]\n"
            "fmadd.ps %[accum1], %[dataValues1], %[weightValues], %[accum1]\n"
            : [ dataValues0 ] "=&f"(dataValues0), [ dataValues1 ] "=&f"(dataValues1), [ srcPtr ] "+&r"(srcPtr),
              [ accum0 ] "+&f"(accum0), [ accum1 ] "+&f"(accum1)
            : [ int8Offsets ] "f"(int8Offsets), [ offsetVector ] "f"(offsetVector), [ scaleVector ] "f"(scaleVector),
              [ weightValues ] "f"(weightValues), [ mask0 ] "r"(mask0), [ mask1 ] "r"(mask1)
            :);
        }

        // store and convert before storing
        __asm__ __volatile__("mov.m.x m0, %[mask0], 0x0\n"
                             "fcvt.f16.ps %[accum0], %[accum0]\n"
                             "fschg.ps %[accum0], %[fp16Offsets](%[dst_ptr0])\n"
                             "mov.m.x m0, %[mask1], 0x0\n"
                             "fcvt.f16.ps %[accum1], %[accum1]\n"
                             "fschg.ps %[accum1], %[fp16Offsets](%[dst_ptr1])\n"
                             :
                             : [ dst_ptr0 ] "r"(dstPtr), [ dst_ptr1 ] "r"(dstPtr + 16), [ accum0 ] "f"(accum0),
                               [ accum1 ] "f"(accum1), [ fp16Offsets ] "f"(fp16Offsets), [ mask0 ] "r"(mask0),
                               [ mask1 ] "r"(mask1)
                             :);
      }

    } else {
      // 8 x FP32 elements on each iteration
      constexpr int64_t vlen = 8;
      for (dim_t i = 0; i < segmentLen; i += vlen) {
        // Set vector length mask
        uint32_t gvlmask = getVMask(segmentLen - i);
        __asm__ __volatile__("mov.m.x m0, %[gvl], 0x0\n" : : [ gvl ] "r"(gvlmask));

        // Clear vector registers that will be used for accumulation
        __asm__ __volatile__("fxor.pi %[accum0], %[accum0], %[accum0]\n" : [ accum0 ] "=&f"(accum0) : :);

        // For every data row in the segment
        for (int64_t j = startOffset; j < endOffset; j++) {

          // Load and broadcast weight
          outType* weightPtr = tWInput + j;
          f32x8 weightValues;
          __asm__ __volatile__("fbc.ps  %[weightValues], 0x0(%[weightPtr])\n"
                               : [ weightValues ] "=&f"(weightValues)
                               : [ weightPtr ] "r"(weightPtr)
                               :);

          // Get the data source pointer
          auto rowIndex = static_cast<dim_t>(IH.raw(j));
          std::array<dim_t, 2> rowCoord = {rowIndex, static_cast<dim_t>(i)};
          auto srcPtr = &DH.at(rowCoord);

          // Load 8 x Int8QTy data elements (srcPtr) and dequantize to FP32
          __asm__ __volatile__("fgb.ps %[dataValues0],   %[int8Offsets](%[srcPtr])\n"
                               // Dequantize: scale * (float)( (¡) input - (int32_t)offset )
                               "fsub.pi %[dataValues0], %[dataValues0], %[offsetVector]\n"
                               // convert to fp32
                               "fcvt.ps.pw %[dataValues0], %[dataValues0]\n"
                               // Mul by scale
                               "fmul.ps %[dataValues0], %[dataValues0], %[scaleVector]\n"
                               : [ dataValues0 ] "=&f"(dataValues0)
                               : [ int8Offsets ] "f"(int8Offsets), [ offsetVector ] "f"(offsetVector),
                                 [ scaleVector ] "f"(scaleVector), [ srcPtr ] "r"(srcPtr)
                               :);

          // Mutiply by weight and Accumulate with previous value
          __asm__ __volatile__("fmadd.ps %[accum0], %[dataValues0], %[weightValues], %[accum0]\n"
                               : [ accum0 ] "+&f"(accum0)
                               : [ dataValues0 ] "f"(dataValues0), [ weightValues ] "f"(weightValues)
                               :);
        }

        // Store
        if (gvlmask >= 0xFF) {
          // Elements remaining >= 8
          __asm__ __volatile__("fswg.ps %[accum0], (%[dst_ptr])\n"
                               :
                               : [ dst_ptr ] "r"(dstPtr), [ accum0 ] "f"(accum0)
                               :);
        } else {
          // Elements remaining is < 8.
          // NOTE: This is necessary because fswg.ps does not support masking yet.
          __asm__ __volatile__("fscwg.ps %[accum0], %[fp32Offsets](%[dst_ptr])\n"
                               :
                               : [ dst_ptr ] "r"(dstPtr), [ accum0 ] "f"(accum0), [ fp32Offsets ] "f"(fp32Offsets)
                               :);
        }
        dstPtr += vlen;
      }
    }
  }
}

template <ElemKind outElK, ElemKind dataElK, ElemKind indexElK>
INLINE_ATTR void fwdLibEmbeddingBagInstFastpath(LibTensor* out, LibTensor* data, LibTensor* weights, LibTensor* indices,
                                                [[maybe_unused]] LibTensor* offsets, bool hasEndOffset, uint64_t flags,
                                                const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  et_assert(data->getElementType() == Int8QTy);
  et_assert(weights->getElementType() == out->getElementType());
  et_assert(indices->getElementType() == offsets->getElementType());
  et_assert((indices->getElementType() == Int64ITy) || (indices->getElementType() == Int32ITy));
  et_assert(reinterpret_cast<uint64_t>(out->getRawDataPointer<uint64_t>()) % 64 == 0);

  (void)hasEndOffset;
  using outType = typename elemKind2elemTy<outElK>::type;
  using inType = typename elemKind2elemTy<dataElK>::type;
  using indicesType = typename elemKind2elemTy<indexElK>::type;

  // If Minion is outside the group assigned to this Node get out.
  size_t minionId = get_minion_id();
  if (minionId < minionOffset) {
    return;
  }

  // Rebase minion ID.
  minionId -= minionOffset;
  // Get number of Minions assigned to this Node.
  size_t activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * activeShires(flags)) : assignedMinions;
  // If Minion is outside the group assigned to this Node get out.
  if (minionId >= activeMinions) {
    return;
  }

  // Work partitioning is based on output tensor 'segments'
  auto numSegments = out->dims()[0];
  // If the EB input data is 1D (x=n), is equivalent to a EB 2D with dims(x=n,y=1)
  dim_t segmentLen = out->ndims() > 1 ? out->dims()[1] : static_cast<dim_t>(1UL);
  dim_t numSegmentsPerMinion = std::max(((numSegments + activeMinions - 1) / activeMinions), 1UL);
  // This minion gets assigned to compute from startSegment to endSegment
  dim_t startSegment = numSegmentsPerMinion * minionId;
  if (startSegment >= numSegments) {
    return;
  }
  auto endSegment = std::min(startSegment + numSegmentsPerMinion, numSegments);

  // Handles
  auto IH = indices->getHandle<indicesType>();
  auto DH = data->getHandle<inType>();
  auto OH = out->getHandle<outType>();
  // Get the sizes to iterate

  // Pre-compute the indices used in gather/scatter operations
  // load gather for quantized Int8QTy data values
  // store scatter for FP16/FP32 accumulated results
  int32_t gather_offsets8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  f32x8 int8Offsets, fp16Offsets, fp32Offsets;
  const int32_t offset = data->getOffset();
  const float scale = data->getScale();
  __asm__ __volatile__("flw.ps  %[int8Offsets], %[gather_offsets8]\n"
                       "fslli.pi  %[fp16Offsets], %[int8Offsets], 1\n"
                       "fslli.pi  %[fp32Offsets], %[fp16Offsets], 1\n"
                       : [ int8Offsets ] "=&f"(int8Offsets), [ fp16Offsets ] "=&f"(fp16Offsets),
                         [ fp32Offsets ] "=&f"(fp32Offsets)
                       : [ gather_offsets8 ] "m"(*(const int32_t(*)[8])gather_offsets8)
                       :);

  // Broadcast scale and offset used in dequantization
  f32x8 offsetVector;
  f32x8 scaleVector;
  __asm__ __volatile__("fbcx.ps %[offsetVector], %[offset]\n"
                       "fbcx.ps %[scaleVector], %[scale]\n"
                       : [ offsetVector ] "=&f"(offsetVector), [ scaleVector ] "=&f"(scaleVector)
                       : [ offset ] "r"(offset), [ scale ] "r"(scale)
                       :);

  // vlen is the number of elements to read/write on each 256-bit stride.
  // e.g: 32x8-bit, 16x16-bit or 8x32-bit
  constexpr int64_t vlen = 32 / sizeof(outType);

  // Main loop
  for (dim_t s = startSegment; s < endSegment; s++) {
    // Get start and end offset of this segment
    auto rowIndex = static_cast<dim_t>(IH.raw(s));

    // Get the destination pointer to write results
    std::array<dim_t, 2> segmentCoord = {s, 0};
    auto dstPtr = &OH.at(segmentCoord);

    // Enable all bits in mask
    __asm__ __volatile__("mov.m.x m0, zero, 0xff\n" : :);

    for (dim_t i = 0; i < segmentLen; i += vlen) {
      // Get the data source pointer
      std::array<dim_t, 2> rowCoord = {rowIndex, static_cast<dim_t>(i)};
      auto srcPtr = reinterpret_cast<char*>(&DH.at(rowCoord));

      if constexpr (outElK == Int8QTy) {
        f32x8 dataValues0;

        // Load 16 x Int8QTy data elements (srcPtr) and dequantize to FP32
        __asm__ __volatile__("flgw.ps %[dataValues0], (%[srcPtr])\n"
                             "fsgw.ps %[dataValues0], (%[dstPtr])\n"
                             : [ dataValues0 ] "=&f"(dataValues0)
                             : [ srcPtr ] "r"(srcPtr), [ dstPtr ] "r"(dstPtr)
                             :);
        // NOTE(1): We assume destination scale is the same.

      } else if constexpr (outElK == Float16Ty) {
        // 16 x FP16 elements on each iteration
        f32x8 dataValues0;
        f32x8 dataValues1;
        // Load 16 x Int8QTy data elements (srcPtr) and dequantize to FP32
        __asm__ __volatile__(
          "fgb.ps %[dataValues0],   %[int8Offsets](%[srcPtr])\n"
          "addi %[srcPtr], %[srcPtr], 8\n"
          "fgb.ps %[dataValues1],   %[int8Offsets](%[srcPtr])\n"
          "addi %[srcPtr], %[srcPtr], 8\n"
          // Dequantize: scale * (float)( (¡) input - (int32_t)offset )
          "fsub.pi %[dataValues0], %[dataValues0], %[offsetVector]\n"
          "fsub.pi %[dataValues1], %[dataValues1], %[offsetVector]\n"
          // convert to fp32
          "fcvt.ps.pw %[dataValues0], %[dataValues0]\n"
          "fcvt.ps.pw %[dataValues1], %[dataValues1]\n"
          // Mul by scale
          "fmul.ps %[dataValues0], %[dataValues0], %[scaleVector]\n"
          "fmul.ps %[dataValues1], %[dataValues1], %[scaleVector]\n"
          : [ dataValues0 ] "=&f"(dataValues0), [ dataValues1 ] "=&f"(dataValues1), [ srcPtr ] "+&r"(srcPtr)
          : [ int8Offsets ] "f"(int8Offsets), [ offsetVector ] "f"(offsetVector), [ scaleVector ] "f"(scaleVector)
          :);

        // Convert to FP16 before storing
        __asm__ __volatile__("fcvt.f16.ps %[dataValues0], %[dataValues0]\n"
                             "fcvt.f16.ps %[dataValues1], %[dataValues1]\n"
                             : [ dataValues0 ] "+&f"(dataValues0), [ dataValues1 ] "+&f"(dataValues1)
                             :
                             :);
        pack_and_global_store_fp16x16(dataValues0, dataValues1, dstPtr);
      } else {
        // 8 x FP32 elements on each iteration
        f32x8 dataValues0;
        // Load 8 x Int8QTy data elements (srcPtr) and dequantize to FP32
        __asm__ __volatile__("fgb.ps %[dataValues0],   %[int8Offsets](%[srcPtr])\n"
                             // Dequantize: scale * (float)( (¡) input - (int32_t)offset )
                             "fsub.pi %[dataValues0], %[dataValues0], %[offsetVector]\n"
                             // convert to fp32
                             "fcvt.ps.pw %[dataValues0], %[dataValues0]\n"
                             // Mul by scale
                             "fmul.ps %[dataValues0], %[dataValues0], %[scaleVector]\n"
                             : [ dataValues0 ] "=&f"(dataValues0)
                             : [ int8Offsets ] "f"(int8Offsets), [ offsetVector ] "f"(offsetVector),
                               [ scaleVector ] "f"(scaleVector), [ srcPtr ] "r"(srcPtr)
                             :);

        // Store
        __asm__ __volatile__("fswg.ps %[dataValues0], (%[dst_ptr])\n"
                             :
                             : [ dst_ptr ] "r"(dstPtr), [ dataValues0 ] "f"(dataValues0)
                             :);
      }
      dstPtr += vlen;
    }
  }
}

//////////// Int8QTy "data" instantiation for EmbeddingBag
// Beware of occupied semantics: "data" from the operator PoV is typically considered a "weight" from dlrm model PoV.

template <ElemKind outElK, ElemKind dataElK, ElemKind indexElK>
INLINE_ATTR void fwdLibEmbeddingBagInst(LibTensor* out, LibTensor* data, LibTensor* weights, LibTensor* indices,
                                        LibTensor* offsets, bool hasEndOffset, uint64_t flags,
                                        const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  (void)flags;
  (void)assignedMinions;

  static_assert(outElK == FloatTy || outElK == Float16Ty);
  static_assert((dataElK == FloatTy && outElK == FloatTy) || (dataElK == Float16Ty && outElK == Float16Ty) ||
                (dataElK == Int8QTy && (outElK == FloatTy)) || (dataElK == Int8QTy && (outElK == Float16Ty)));

  using outType = typename elemKind2elemTy<outElK>::type;
  using dataType = typename elemKind2elemTy<dataElK>::type;
  using outGlobalType = typename std::conditional<outElK == ElemKind::FloatTy, uint32_t, uint16_t>::type;
  using indicesType = typename elemKind2elemTy<indexElK>::type;

  auto minionId = get_minion_id();
  //  FIXME: just minon 0 does some work at the moment.
  if ((minionId - minionOffset) != 0) {
    return;
  }

  auto IH = indices->getHandle<indicesType>();
  auto OFFH = offsets->getHandle<indicesType>();

  // If an end offset is present to mark the end of the last segment then this
  // must be subtracted to get the correct number of segments
  size_t segments = hasEndOffset ? offsets->dims()[0] - 1 : offsets->dims()[0];
  size_t numIndices = indices->dims()[0];

  size_t lineSize = data->size() / data->dims()[0];

  auto DH = data->getHandle<dataType>();
  auto WH = weights->getHandle<outType>();
  auto OH = out->getHandle<outType>();

  // clean the ouput-tensor. We should not clean the padding, so using
  // explicit indexed writes.
  // et_assert(ndims = 2)
  for (dim_t d0 = 0; d0 < out->dims()[0]; d0++) {
    for (dim_t d1 = 0; d1 < out->dims()[1]; d1++) {
      std::array<dim_t, 2> pos = {d0, d1};
      outType zeroF = 0;
      auto address = &OH.at(pos);

      if constexpr (outElK == Float16Ty) {
        atomic_store_global_16(reinterpret_cast<outGlobalType*>(address), *reinterpret_cast<outGlobalType*>(&zeroF));
      } else {
        atomic_store_global_32(reinterpret_cast<outGlobalType*>(address), *reinterpret_cast<outGlobalType*>(&zeroF));
      }
    }
  }

  dim_t curIdx = 0;
  for (dim_t i = 0; i < segments; i++) {
    dim_t start = OFFH.raw(i);
    dim_t end;
    if (!hasEndOffset) {
      // Note that in this case we have to use numIndices to find the end of
      // the last segment. This is an issue though because it relies on knowing
      // the total length of the indices tensor which may not be possible.
      // Future implementations of this operator should always give an end
      // offset so eventually this case should be removed.
      end = i == segments - 1 ? numIndices : OFFH.raw(i + 1);
    } else {
      end = OFFH.raw(i + 1);
    }
    if (start == end) {
      continue;
    } else if (start > end) {
      break;
    }
    for (dim_t j = start; j < end; j++) {
      float weight;
      if constexpr (outElK == Float16Ty) {
        auto tmp = WH.raw(curIdx);
        convertFp16ToFp32(tmp, weight);
      } else {
        weight = WH.raw(curIdx);
      }
      size_t dataLine = IH.raw(curIdx);
      curIdx++;

      // prepare data pos for dataLine.
      std::array<dim_t, 2> inPos = {dataLine, 0};

      // prepare and output pos for ith-line.
      std::array<dim_t, 2> outPos = {i, 0};

      for (dim_t k = 0; k < lineSize; k++) {
        float currVal, newVal;
        // Read from and write to outAddress
        auto outAddress = reinterpret_cast<outGlobalType*>(&OH.at(outPos));

        if constexpr (outElK == Float16Ty) {
          // FP16
          outType fp16tmp;
          float fp32tmp;

          // Load current value and convert to FP32
          auto fp16Value = atomic_load_global_16(outAddress);
          convertFp16ToFp32(fp16Value, currVal);

          // Load data value and convert to FP32 before use.
          dataType value = DH.at(inPos);
          if constexpr (dataElK == Int8QTy) {
            newVal = currVal + dequantize(value, data->getScale(), data->getOffset()) * weight;
          } else {
            convertFp16ToFp32(value, fp32tmp);
            // Compute new value
            newVal = currVal + fp32tmp * weight;
          }
          // Write
          convertFp32ToFp16(newVal, fp16tmp);
          atomic_store_global_16(outAddress, *(reinterpret_cast<outGlobalType*>(&fp16tmp)));

        } else {
          // FP32

          // Load current value
          auto rawValue = atomic_load_global_32(outAddress);
          currVal = *reinterpret_cast<outType*>(&rawValue);

          // Load data value
          dataType value = DH.at(inPos);

          if constexpr (dataElK == Int8QTy) {
            // Data value is Int8QTy. Dequantize value to FP32 before use.
            newVal = currVal + dequantize(value, data->getScale(), data->getOffset()) * weight;
          } else {
            newVal = currVal + value * weight;
          }

          // Write
          atomic_store_global_32(outAddress, *(reinterpret_cast<outGlobalType*>(&newVal)));
        }

        // increment innder dims
        inPos[1]++;
        outPos[1]++;
      }
    }
  }
}

} // namespace inlining
} // namespace dnn_lib

#undef PACK_AND_GLOBAL_STORE_8_FP16
#undef PACK_AND_GLOBAL_STORE_16_FP16
#undef PACK_AND_STORE_8_FP16
#undef PACK_AND_STORE_16_FP16

#endif // _EMBEDDING_BAG_INST_H_

