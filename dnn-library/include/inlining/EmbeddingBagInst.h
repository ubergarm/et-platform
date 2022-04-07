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

#include "Float16.h"
#include "utils.h" // From include/internal path
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

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
                         "fmvz.x.ps %[stReg1], " #FP_REG_ ", 1\n     "                                                 \
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

template <ElemKind elK>
inline __attribute((always_inline)) void
embeddingBagsTailVectorized(uintptr_t minionCurrIndex, uintptr_t currSegmentStart, uintptr_t currSegmentEnd,
                            uint8_t* tAInput, int64_t* indices, uintptr_t dataRowPitch, uintptr_t dataRowGroupOffset,
                            uint8_t* tWInput, uint8_t* dst_ptr, bool destAlignedVreg) {

  constexpr bool float32Dst = (elK == FloatTy);
  constexpr bool float16Dst = (elK == Float16Ty);

  static_assert(elK == FloatTy || elK == Float16Ty);

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

    uint8_t *data_ptr   = tAInput + indices[currIndex] * dataRowPitch
                                  + dataRowGroupOffset;
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
    if (destAlignedVreg) {
      // Store accumulated results.
      __asm__ __volatile__("fswg.ps f0, (%[dst_ptr])\n" : : [ dst_ptr ] "r"(dst_ptr) : "f0");
    } else {
      // Store accumulated results.
      __asm__ __volatile__ (
        "fscwg.ps f0, f20(%[dst_ptr])\n"
        :
        : [dst_ptr] "r" (dst_ptr)
        : "f0", "f20"
      );
    }
  } else {
    __asm__ __volatile__ (
      "fcvt.f16.ps f0, f0\n"
      :
      :
      : "f0"
    );
    if (destAlignedVreg) {
      PACK_AND_GLOBAL_STORE_8_FP16(f0, dst_ptr);
    } else {
      __asm__ __volatile__ (
        "fschg.ps f0, f20(%[dst_ptr])\n"
        :
        : [dst_ptr] "r" (dst_ptr)
        : "f0", "f20"
      );
    }
  }
}

template <ElemKind elK>
inline __attribute((always_inline)) void fwdLibEmbeddingBagInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                                                LibTensor* in3T, LibTensor* in4T, bool hasEndOffset,
                                                                uint64_t flags, const uint32_t minionOffset = 0,
                                                                const uint32_t assignedMinions = 0) {

  constexpr bool float32Dst = (elK == FloatTy);
  constexpr bool float16Dst = (elK == Float16Ty);

  static_assert(elK == FloatTy || elK == Float16Ty);
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

  assert(in1T->getElementType() == outT->getElementType());
  assert((in1T->getElementType() == FloatTy) || (in1T->getElementType() == Float16Ty));
  assert((in3T->getElementType() == Int64ITy) && (in3T->getElementType() == in4T->getElementType()));

  auto offH = in4T->getHandle<int64_t>();

  auto tOutput = outT->getRawDataPointer<uint8_t>();
  auto tAInput = in1T->getRawDataPointer<uint8_t>();
  auto tWInput = in2T->getRawDataPointer<uint8_t>();
  auto indices = in3T->getRawDataPointer<int64_t>();

  const dim_t segments = hasEndOffset ? (in4T->dims()[0] - 1) : in4T->dims()[0];
  const dim_t numIndices = in3T->dims()[0];

  // Compute the number of elements per data row (first tensor dimension).
  const uintptr_t dstRowElemSize = outT->dims()[1];  
  
  // Assign work to Minions :
  //
  // NOTE: Each Minion gets assigned at least an output cache line for
  // historical reasons although Global stores are in use.
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
  const uintptr_t dstVRegElems = 8; // SIMD/Vector length  in elements
  const uintptr_t dstGroupElems = CACHE_LINE_BYTES /  elemSize;
  const uintptr_t dstGroupVRegs = CACHE_LINE_BYTES / (elemSize * 8);

  // Compute the number of groups per output row (rounded up).
  const uintptr_t dstRowGroups = ((dstRowElemSize - 1) / dstGroupElems) + 1;

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
  if (minionWorkUnits == 0) {
    return;
  }

  // Computes if the destination is correctly aligned to vReg and contiguous stores
  // can be used isntead of scatters.
  // Need both the dest starting address being VReg aligned as well as the pitch for
  // the smallest dimension
  // The padding must be touchable or the number of elements multiple of Vreg size
  const auto outputStrideZeroBytes = outT->strides()[0] * elemSize;
  bool destAlignedVreg = (((uint64_t)tOutput % VREG_BYTES) == 0) && ((outputStrideZeroBytes % VREG_BYTES) == 0) &&
                         (!outT->getUntouchable() || (((uint64_t)outT->dims()[1] % dstVRegElems) == 0));

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
    if (start >= end) {
      emptySegment = true;
    } else {
      emptySegment = false;
    }
    currSegmentStart = start;
    currSegmentEnd   = end;
    return emptySegment;
  };

  if (offH.raw(0) == 0) {
    // fast path, offests start at 0, we can have a direct access to minionFirstIndex
    minionFirstIndex = offH.raw(minionFirstSegment);
    (void)getNextSegment(minionFirstSegment);
  } else {
    // slow path, offests do not start at 0, offets need traveral to get to minionFirstIndex
    bool minionEmptySegment = false;
    for (uintptr_t s = 0; s <= minionFirstSegment; s++) {
      if (not minionEmptySegment)
        minionFirstIndex += (currSegmentEnd - currSegmentStart);
      minionEmptySegment = getNextSegment(s);
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
        uint8_t *data_ptr   = tAInput + (  indices[currIndex] * in1T->strides()[0]
                                         + minionCurrRowGroup * dstGroupElems) * elemSize;
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
            tAInput + (indices[currIndex + 1] * in1T->strides()[0] + minionCurrRowGroup * dstGroupElems) * elemSize;
          // Prefetch next index of current segment
          __asm__ __volatile__("ld          x0, (%[data_ptr_next])\n" : : [ data_ptr_next ] "r"(data_ptr_next) :);
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
        if (destAlignedVreg) {
          // Store accumulated results.
          __asm__ __volatile__("fswg.ps f0,  (%[dst_ptr])\n"
                               "addi    %[dst_ptr], %[dst_ptr], 32\n"
                               "fswg.ps f1,  (%[dst_ptr])\n"
                               "addi    %[dst_ptr], %[dst_ptr], 32\n"
                               : [ dst_ptr ] "+&r"(dst_ptr)
                               :
                               : "f0", "f1");
        } else {
          __asm__ __volatile__ (
            "fscwg.ps    f0, f20(%[dst_ptr])\n"
            "addi        %[dst_ptr], %[dst_ptr], 32\n"
            "fscwg.ps    f1, f20(%[dst_ptr])\n"
            "addi        %[dst_ptr], %[dst_ptr], 32\n"
            : [dst_ptr] "+&r" (dst_ptr)
            :
            : "f0", "f1", "f20"
          );
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

        if (destAlignedVreg) {
          PACK_AND_GLOBAL_STORE_16_FP16(f0, f1, dst_ptr);
          PACK_AND_GLOBAL_STORE_16_FP16(f2, f3, dst_ptr);
        } else {
          __asm__ __volatile__ (
            "fschg.ps     f0, f20(%[dst_ptr])\n"
            "addi        %[dst_ptr], %[dst_ptr], 16\n"
            "fschg.ps     f1, f20(%[dst_ptr])\n"
            "addi        %[dst_ptr], %[dst_ptr], 16\n"
            "fschg.ps     f2, f20(%[dst_ptr])\n"
            "addi        %[dst_ptr], %[dst_ptr], 16\n"
            "fschg.ps     f3, f20(%[dst_ptr])\n"
            "addi        %[dst_ptr], %[dst_ptr], 16\n"
            : [dst_ptr] "+&r" (dst_ptr)
            :
            : "f0", "f1", "f2", "f3", "f20"
          );
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

        getNextSegment(minionCurrSegment);
        dst_ptr = tOutput + minionCurrSegment * outputStrideZeroBytes + minionCurrRowGroup * dstGroupElems * elemSize;
      }
    } else {
      // Initialize mask to clear upper bytes from input load
      __asm__ __volatile__ (
        "mov.m.x m0, zero, 0xff\n"
      );

      for (uintptr_t k = 0; k < (dstRowTailVRegs - 1); k++) {
        embeddingBagsTailVectorized<elK>(
          minionCurrIndex, currSegmentStart, currSegmentEnd, tAInput, indices, in1T->strides()[0] * elemSize,
          (minionCurrRowGroup * dstGroupElems + k * dstVRegElems) * elemSize, tWInput, dst_ptr, destAlignedVreg);
        dst_ptr += dstVRegElems * elemSize;
      }

      // Set mask for last VReg in group.
      __asm__ __volatile__ (
        "mov.m.x m0, %[tail_mask], 0x0\n"
        :
        : [tail_mask] "r" (dstRowTailVRegMask)
      );

      embeddingBagsTailVectorized<elK>(
        minionCurrIndex, currSegmentStart, currSegmentEnd, tAInput, indices, in1T->strides()[0] * elemSize,
        (minionCurrRowGroup * dstGroupElems + (dstRowTailVRegs - 1) * dstVRegElems) * elemSize, tWInput, dst_ptr,
        destAlignedVreg);

      minionCurrIndex += (currSegmentEnd - currSegmentStart);

      // Move from row tail to next row.
      minionCurrSegment++;
      minionCurrRowGroup = 0;

      getNextSegment(minionCurrSegment);

      dst_ptr = tOutput + minionCurrSegment * outputStrideZeroBytes + minionCurrRowGroup * dstGroupElems * elemSize;
    }
  }
}

} // namespace inlining
} // namespace dnn_lib

#undef PACK_AND_GLOBAL_STORE_8_FP16
#undef PACK_AND_GLOBAL_STORE_16_FP16

#endif // _EMBEDDING_BAG_INST_H_

