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

#ifndef _SPARSE_TO_DENSE_MASK_INST_H
#define _SPARSE_TO_DENSE_MASK_INST_H

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

  template <ElemKind elK, size_t N>
inline void fwdLibSparseToDenseMaskInst(LibTensor* outT, LibTensor* in1T,
                                        LibTensor* in2T, LibTensor* in3T,
                                        LibTensor* in4T,
                                        const std::array<size_t, N> mask,
                                        uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0
                                        ) {

    //    using srcType = typename elemKind2elemTy<elK>::type;
    if (get_minion_id() != minionOffset) return;
    
  /* maintain compatibility through the new Iface Libtensor */
  /* out--> dest in2T->val in3T->dft in1T->idx in4T->len*/
  void* pdst = outT->getRawDataPointer<void>();
  void* pdata = in2T->getRawDataPointer<void>();
  void* pdefault = in3T->getRawDataPointer<void>();
  //  unsigned int pdefaultSize = in3T->size();
  unsigned int pLengthsSize = in4T->size();
  // Addresser<elK> tOutput(pdst, scale[2], offset[2]);
  Addresser<elK> tOutput(pdst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tAInput(pdata, scale[0], offset[0]);
  const Addresser<elK> tAInput(pdata, in2T->getScale(), in2T->getOffset());
  // const Addresser<elK> tDefVInput(pdefault, scale[1], offset[1]);
  const Addresser<elK> tDefVInput(pdefault, in3T->getScale(), in3T->getOffset());
  // long long *indices = (long long *)pindices;
  size_t *indices = in1T->getRawDataPointer<size_t>();
  // int32_t *lengths = (int32_t *)plengths;
  int32_t *lengths = in4T->getRawDataPointer<int32_t>();

  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch = outT->strides().data();
  
  // First un-processed index-value pair.
  size_t posIn = 0;
  // Beginning of output block for first unprocessed batch.
  size_t byteoffsetOut = 0;
  // Lengths can be scalar, which means that all pairs belong to one batch.
  size_t numBatches = (pLengthsSize == 1) ? 1 : lengths[0];
  // Go to the next batch
  size_t advanceBatch = (pLengthsSize == 1) ? 0 : dstPitch[0];
  // Go to the next position inside batch (row, column..)
  size_t advanceInBatch = (pLengthsSize == 1) ? dstPitch[0] : dstPitch[1];

  // Position of idx in the mask
  size_t j = 0;
  uint64_t srcAddr, srcAddrUp, dstAddr;
  for (size_t batch = 0; batch < numBatches; batch++) {
    // Fill everything with maskSize copies of defaultValue.
    for (size_t i = 0; i < mask.size(); i++) {
      srcAddr = 0;
      srcAddrUp = //pdefaultSize;
      dstAddr = byteoffsetOut + advanceInBatch * i;
      auto val = tDefVInput[0];
      for (uint64_t addr = srcAddr, cnt = 0; addr < srcAddrUp; addr++, cnt++) {
        val = tDefVInput[addr];
        tOutput[dstAddr + cnt] = val;
      }
    }
    // Go through input pairs and find matches.
    for (int32_t i = 0; i < lengths[batch]; i++, posIn++) {
      auto idx = indices[posIn];
      // Search the mask
      for (j = 0; j < mask.size(); j++) {
        if (mask[j] == idx) {
          break;
        }
      }
      // Skip if ID is not present in the mask.
      if (j == mask.size())
        continue;

      srcAddr = posIn * advanceInBatch;
      srcAddrUp = (posIn + 1) * advanceInBatch;
      dstAddr = byteoffsetOut + advanceInBatch * j;
      auto val = tAInput[0];
      for (uint64_t addr = srcAddr, cnt = 0; addr < srcAddrUp; addr++, cnt++) {
        val = tAInput[addr];
        tOutput[dstAddr + cnt] = val;
      }
    }

    byteoffsetOut += advanceBatch;
  }
}

// Assumptions for the SparseToDenseMaskInst threaded version:
// (1) The pmask vector size (pMaskSize) has the same length as the second dimension of the output tensor.
// (2) The dimensions and pitches of the pdefault tensor are the ones of a batch of the data tensor.

  template <ElemKind elK, size_t N>
inline void fwdLibSparseToDenseMaskInstThreaded(LibTensor* outT, LibTensor* in1T,
                                                LibTensor* in2T, LibTensor* in3T,
                                                LibTensor* in4T,
                                                const std::array<size_t, N> mask,
                                                uint64_t flags,
                                                const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;
  
  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */
  /* out--> dest in2T->val in3T->dft in1T->idx in4T->len*/
  void *pdst = outT->getRawDataPointer<void>();
  void *pdata = in2T->getRawDataPointer<void>();
  void *pdefault = in3T->getRawDataPointer<void>();
  
  //  unsigned int pdefaultSize = in2T->size();
  unsigned int pLengthsSize = in4T->size();
  
  // Addresser<elK> tOutput(pdst, scale[2], offset[2]);
  Addresser<elK> tOutput(pdst, outT->getScale(), outT->getOffset());
  // const Addresser<elK> tAInput(pdata, scale[0], offset[0]);
  const Addresser<elK> tAInput(pdata, in2T->getScale(), in2T->getOffset());
  // const Addresser<elK> tDefVInput(pdefault, scale[1], offset[1]);
  const Addresser<elK> tDefVInput(pdefault, in3T->getScale(), in3T->getOffset());
  // long long *indices = (long long *)pindices;
  size_t *indices = in1T->getRawDataPointer<size_t>();
  // int32_t *lengths = (int32_t *)plengths;
  int32_t *lengths = in4T->getRawDataPointer<int32_t>();

  // unsigned int *dstIndex = (unsigned int *)pdstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dataIndex = (unsigned int *)pdataDims;
  const dim_t *dataIndex = in2T->dims().data();
  // unsigned int *dstPitch = (unsigned int *)pdstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *dataPitch = (unsigned int *)pdataPitches;
  const dim_t *dataPitch = in2T->strides().data();

  unsigned int pdstDimNum = static_cast<unsigned int>(outT->ndims());
  unsigned int pdataDimNum = static_cast<unsigned int>(in2T->ndims());
  
  unsigned int numElemsDst = dstPitch[0]*dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, pdst);
  if (maxRead == 0)
    return;

  unsigned int coordOut[pdstDimNum];
  unsigned int last_non_zero_coord;
  getNonPaddingCoordinates(coordOut, initialAddr, pdstDimNum, dstPitch, dstIndex, last_non_zero_coord);

  uint64_t offsetOut = 0;
  for (unsigned int i = 0; i < last_non_zero_coord; i++) {
    offsetOut += dstPitch[i]*coordOut[i];
  }

  unsigned int batchCount = offsetOut/dstPitch[0];
  unsigned int mod = offsetOut - batchCount*dstPitch[0];
  unsigned int semiBatchCount = mod/dstPitch[1];
  unsigned int offsetIn = mod - semiBatchCount*dstPitch[1];

// The default value tensor's indexes and pitches can be obtained from the data tensor, as a consequence
// of assumption (2) listed above.
  unsigned int pdefDimNum = pdataDimNum - 1;
  unsigned int defPitch[pdefDimNum];
  unsigned int defIndex[pdefDimNum];
  for (unsigned int i = 0; i < pdefDimNum; i++) {
    defPitch[i] = dataPitch[i + 1];
    defIndex[i] = dataIndex[i + 1];
  }

  unsigned int coordIn[pdefDimNum]; // Coordinates in the default value tensor (or an input batch).
  getNonPaddingCoordinates(coordIn, offsetIn, pdefDimNum, defPitch, defIndex, last_non_zero_coord);
  offsetIn = 0;
  for (unsigned int i = 0; i < last_non_zero_coord; i++) {
    offsetIn += defPitch[i]*coordIn[i];
  }

  unsigned int firstIdx = 0;
  for (unsigned int i = 0; i < batchCount; ++i) firstIdx += lengths[i];
  unsigned int lastIdx = firstIdx + lengths[batchCount];

  unsigned int idx = mask[semiBatchCount];
  bool defaultVal = true;
  unsigned int j;

  for (j = firstIdx; j < lastIdx; j++) {
    if (indices[j] == idx) {
      defaultVal = false;
      break;
    }
  }

  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  bool doneIn = false;

  while (!done && (offsetOut < posMax)) {

    if (defaultVal) tOutput[offsetOut] = tDefVInput[offsetIn];
    else tOutput[offsetOut] = tAInput[j*dataPitch[0] + offsetIn];

    done = getOffsets(pdstDimNum, coordOut, offsetOut, dstIndex, dstPitch);
    doneIn = getOffsets(pdefDimNum, coordIn, offsetIn, defIndex, defPitch);

    if (doneIn) {
      doneIn = false;
      offsetIn = 0;
      for (unsigned int i = 0; i < pdefDimNum; ++i) {
        coordIn[i] = 0;
      }
      ++semiBatchCount;
      if ((semiBatchCount == mask.size()) and // Assumption (1): pMaskSize = dstIndex[1].
          (batchCount < pLengthsSize-1)) {
        semiBatchCount = 0;
        ++batchCount;
        firstIdx = lastIdx;
        lastIdx += lengths[batchCount];
      }
      idx = mask[semiBatchCount];
      defaultVal = true;
      for (j = firstIdx; j < lastIdx; j++) {
        if (indices[j] == idx) {
          defaultVal = false;
          break;
        }
      }
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _SPARSE_TO_DENSE_MASK_INST_H
