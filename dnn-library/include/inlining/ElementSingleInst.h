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

#ifndef _ELEMENT_SINGLE_INST_H_
#define _ELEMENT_SINGLE_INST_H_

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

template <ElemKind dstElK, ElemKind srcElK, typename opType>
inline void fwdLibElementSingleInst(LibTensor* outT, LibTensor* inT,
                                    uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;
  using srcType = typename elemKind2elemTy<srcElK>::type;
  using dstType = typename elemKind2elemTy<dstElK>::type; 
  const auto aSrcT1 = inT->getHandle<srcType>();
  auto aDstT = outT->getHandle<dstType>();

  const dim_t *srcIndex = inT->dims().data();
  
  uint8_t srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  unsigned int eBatchDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};

  for (size_t i = 0; i < srcDimNum; i++) {
    eBatchDims[i] = srcIndex[i];
  }

  Operator<srcType, srcType, dstType, opType> op;
  dim_array_t d = {0,0,0,0,0,0};
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n");
  // We can use this loop for all shapes.
  for (d[0] = 0; d[0] < eBatchDims[0]; d[0]++) {
    for (d[1] = 0; d[1] < eBatchDims[1]; d[1]++) {
      for (d[2] = 0; d[2] < eBatchDims[2]; d[2]++) {
        for (d[3] = 0; d[3] < eBatchDims[3]; d[3]++) {
          for (d[4] = 0; d[4] < eBatchDims[4]; d[4]++) {
            for (d[5] = 0; d[5] < eBatchDims[5]; d[5]++) {
              op.doOp(aDstT, aSrcT1, d);
            }
          }
        }
      }
    }
  }
}

template <ElemKind dstElK, ElemKind srcElK, typename opType>
inline void fwdLibElementSingleInstThreaded(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                            const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<srcElK>::type;
  using dstType = typename elemKind2elemTy<dstElK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  void *dst = outT->getRawDataPointer<void>();
  const auto aSrcT1 = inT->getHandle<srcType>();
  auto aDstT = outT->getHandle<dstType>();

  const dim_t *actIndex = inT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *actPitch = inT->strides().data();

  uint8_t srcDimNum = static_cast<unsigned int>(inT->ndims());

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dst);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  Operator<srcType, srcType, dstType, opType> op;
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  __asm__ __volatile__("mov.m.x m0, zero, 0x1 \n");
  while (!done && (offsetOut < posMax)) {
    op.doOp(aDstT, aSrcT1, offsetOut, offsetIn);
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }

  /* maintain compatibility through the new Iface Libtensor */
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <ElemKind dstElK, ElemKind srcElK, typename opType>
inline void fwdLibElementSingleInstVectorized(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                              const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using dstType = typename elemKind2elemTy<dstElK>::type;


  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;


  /* maintain compatibility through the new Iface Libtensor */
  void* dstT = outT->getRawDataPointer<void>();
  void* srcT1 = inT->getRawDataPointer<void>();
  
  const dim_t *actIndex = inT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *actPitch = inT->strides().data();
  
  uintptr_t dstAddr = (uintptr_t)dstT;
  uintptr_t srcAddr = (uintptr_t)srcT1;

  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());

  
  Operator<Addresser<srcElK>, Addresser<srcElK>, Addresser<dstElK>, opType> op;

  unsigned int numElemsDst = dstPitch[0] * actIndex[0];
 
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);
  if (maxRead == 0)
    return;

  unsigned int coord[srcDimNum];
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, actIndex,
                           k);

  uint64_t offsetIn = 0;
  uint64_t offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += actPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  unsigned int lastDim = srcDimNum - 1;

  int32_t gatherValues[] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (unsigned int i = 0; i < 8; i++) {
      gatherValues[i] = i * typeSize;
  }

  unsigned int maxRow = (srcDimNum > 1) ? (posMax / dstPitch[lastDim - 1]) : 0;
  unsigned int elementsInRow, registersInRow, res;
  uint8_t mask;
  bool firstRow = true;
  bool midRow = false;
  bool lastRow = false;
  lastDim += (srcDimNum == 1);
  coord[0] *= (srcDimNum != 1);

  while (!done && (offsetOut < posMax)) {
    if (firstRow && coord[lastDim - 1] != maxRow) {
      elementsInRow = actIndex[lastDim] - coord[lastDim];
    } else if (coord[lastDim - 1] == maxRow) {
      lastRow = true;
      elementsInRow = posMax - offsetOut;
    } else {
      elementsInRow = actIndex[lastDim];
    }
    if (firstRow || lastRow || !midRow) { // cases where variable update is needed.
      registersInRow = elementsInRow / 8;
      res = elementsInRow - registersInRow * 8;
      mask = ((1 << res) - 1);
      if (!firstRow) midRow = true;
    }
    firstRow = false;
    srcAddr += offsetIn * typeSize;
    dstAddr += offsetOut * typeSize;
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n");

    int32_t offset[] = {inT->getOffset(), outT->getOffset()};
    float scale[] =  {inT->getScale(), outT->getScale()};
      
    unsigned int cnt = 0;
    while(cnt < registersInRow) {
      /* review in implementation sw-2429 to tacke out scale and offset from params and set what it is necessary*/
      op.doOpVect(gatherValues, srcAddr, dstAddr, scale, offset);
      cnt++;
      srcAddr += 8 * typeSize;
      dstAddr += 8 * typeSize;
    }
    if (res > 0) {
      __asm__ __volatile__("mov.m.x m0, %[mask], 0 \n" : : [ mask ] "r"(mask) :);
      /* review in implementation sw-2429 to tacke out scale and offset from params and set what it is necessary*/
      op.doOpVect(gatherValues, srcAddr, dstAddr, scale, offset);
    }
    if (lastRow)
      return;

    dstAddr = (uintptr_t)dstT;
    srcAddr = (uintptr_t)srcT1;
    
    offsetIn -= coord[lastDim] * actPitch[lastDim];
    offsetOut -= coord[lastDim] * dstPitch[lastDim];
    coord[lastDim] = 0;
    done = getOffsets(lastDim , coord, offsetIn, offsetOut, actIndex,
                      actPitch, dstPitch);
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}


  ////////////////////////////////////////////////////////////////////////////////
  // instances for particular instructions calling the above functions
  ////////////////////////////////////////////////////////////////////////////////

  // instances where src and dst can have different types  
#define ELT_SINGLE_INSTANCE_1K(name, version, dstElK, srcElK)                                                             \
  template <ElemKind elK>                                                                                                 \
  inline void fwdLib ## name ## Inst                                                                                      \
  (LibTensor* outT, LibTensor* inT,uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) { \
    inlining::fwdLibElementSingleInst ## version <dstElK, srcElK, name>(outT, inT, flags, minionOffset, assignedMinions); \
  } 

  // instances where src and dst can have different types
#define ELT_SINGLE_INSTANCE_2K(name, version, dstElK, srcElK)                                                             \
  template <ElemKind elK1, ElemKind elK2>                                                                                 \
  inline void fwdLib ## name ## Inst                                                                                      \
  (LibTensor* outT, LibTensor* inT,uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) { \
    inlining::fwdLibElementSingleInst ## version <dstElK, srcElK, name>(outT, inT, flags, minionOffset, assignedMinions); \
  } 
  

  
  // Log has vectorized as generic version
  ELT_SINGLE_INSTANCE_2K(ElementLog, Vectorized, elK1, elK2)
  
  // others have threaded as generic version (vectorized not implemented) 
  ELT_SINGLE_INSTANCE_1K(ElementExp, Threaded, elK, elK)
  ELT_SINGLE_INSTANCE_1K(ElementIsNaN, Threaded, BoolTy, elK)
  ELT_SINGLE_INSTANCE_1K(Sigmoid, Threaded, elK, elK)
  ELT_SINGLE_INSTANCE_1K(Tanh, Threaded, elK, elK)

  
#undef ELT_SINGLE_INSTANCE_1K
#undef ELT_SINGLE_INSTANCE_2K
 

  
} // namespace inlining

} // namespace dnn_lib

#endif // _ELEMENT_SINGLE_INST_H_




