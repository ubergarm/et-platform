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

#ifndef _GATHER_RANGES_INST_H_
#define _GATHER_RANGES_INST_H_

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

template <typename srcType, typename indexType>
inline void fwdLibGatherRangesInst(LibTensor* inT, LibTensor* outT,
                                   LibTensor* out2T, LibTensor* rangesT) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  /* maintain compatibility through the new Iface Libtensor */

  void* srcT = reinterpret_cast<void*>(inT->getUnsafePtr());
  void* dstT = reinterpret_cast<void*>(outT->getUnsafePtr());
  void* prangesT = reinterpret_cast<void*>(rangesT->getUnsafePtr());
  void* dst2T = reinterpret_cast<void*>(out2T->getUnsafePtr());
  
  
  // Addresser<srcType> tOutput(dstT, scale[3], offset[3]);
  // const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  // Addresser<indexType> tRanges(prangesT, scale[1], offset[1]);
  // Addresser<indexType> tLengths(dst2T, scale[2], offset[2]);
  Addresser<srcType> tOutput(dstT, outT->dbggetscale(), outT->dbggetoffset());
  const Addresser<srcType> tInput(srcT, inT->dbggetscale(), inT->dbggetoffset());
  Addresser<indexType> tRanges(prangesT, rangesT->dbggetscale(), rangesT->dbggetoffset());
  Addresser<indexType> tLengths(dst2T, out2T->dbggetscale(), out2T->dbggetoffset());

  
  // unsigned int *srcIndex = (unsigned int *)srcDims;
  dim_t srcIndex[max_tensor_dimensions] = {0,};
  inT->dims(srcIndex);
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  dim_t dstIndex[max_tensor_dimensions] = {0,};
  outT->dims(dstIndex); 
  // unsigned int *rangesIndex = (unsigned int *)prangesDims;
  dim_t rangesIndex[max_tensor_dimensions] = {0,};
  rangesT->dims(rangesIndex);  
  // unsigned int *lenIndex = (unsigned int *)dst2Dims;
  dim_t lenIndex[max_tensor_dimensions] = {0,};
  out2T->dims(lenIndex);

  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  dim_t srcPitch[max_tensor_dimensions] = {0,};
  inT->dbgcpypitches(srcPitch);
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  dim_t dstPitch[max_tensor_dimensions] = {0,};
  outT->dbgcpypitches(dstPitch);  
  // unsigned int *rangesPitch = (unsigned int *)prangesPitches;
  dim_t rangesPitch[max_tensor_dimensions] = {0,};
  rangesT->dbgcpypitches(rangesPitch);  
  // unsigned int *lenPitch = (unsigned int *)dst2Pitches;
  dim_t lenPitch[max_tensor_dimensions] = {0,};
  out2T->dbgcpypitches(lenPitch);

  
  // Offset into the output tensor that keeps track of where to start
  // copying data.
  uint64_t outP = 0;

  // unsigned dataElementSize = dataTy.getElementSize();
  indexType numExamples = rangesIndex[0];
  indexType exampleSize = rangesIndex[1];

  // Keep track of the total number of elements gathered across all
  // examples for a sanity check later.
  size_t grandTotalLen = 0;

  // For each example in ranges:
  for (indexType example = 0; example < numExamples; example++) {
    // Keep a running total of the lengths of all ranges in this example
    // to record into lengthsT once the entire example is processed.
    indexType totalLen = 0;

    // For each range in the example:
    for (indexType range = 0; range < exampleSize; range++) {
      // Get the start index and range length.
      indexType startIdx =
          tRanges[example * rangesPitch[0] + range * rangesPitch[1]];
      indexType len = tRanges[example * rangesPitch[0] +
                              range * rangesPitch[1] + 1 * rangesPitch[2]];

      // Add the length of this current range to the example length counter.
      totalLen += len;

      // Copy the specified data to outT.
      uint64_t srcAddr = startIdx * srcPitch[0];
      uint64_t srcAddrUp = (startIdx + len) * srcPitch[0];

      auto val = tInput[0];
      for (uint64_t i = srcAddr, j = 0; i < srcAddrUp; i++, j++) {
        val = tInput[i];
        tOutput[outP + j] = val;
      }

      // Advance the offset into outT.
      outP += len * dstPitch[0];
    }

    // Record the total number of elements gathered for the example in
    // lengthsT.
    tLengths[example * lenPitch[0]] = totalLen;

    // Add the total length of the entire example to the grand total.
    grandTotalLen += static_cast<size_t>(totalLen);
  }

  // Make sure that number of elements written to outT is equal to the
  // total of all elements in lengthsT.
  // assert(grandTotalLen == (outP / dstPitch[0]));
}


// The range tensor has dimensions n x m x 2, where n is the number of examples and m is the number of
// ranges per example. For any pair (i,j), the element ranges[i,j,0] is the source tensor batch number from
// which the copy will start, and the element ranges[i,j,1] is the length of the copy, that is, the amount
// of batches of the source tensor that will be copied.

template <typename srcType, typename indexType>
inline void fwdLibGatherRangesInstThreaded(LibTensor* inT, LibTensor* outT,
                                           LibTensor* out2T, LibTensor* rangesT,
                                           uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = MIN_PER_SHIRE*ACTIVE_SHIRES;
  if (minionId >= activeMinions) return;

  /* maintain compatibility through the new Iface Libtensor */

  void* srcT = reinterpret_cast<void*>(inT->getUnsafePtr());
  void* dstT = reinterpret_cast<void*>(outT->getUnsafePtr());
  void* prangesT = reinterpret_cast<void*>(rangesT->getUnsafePtr());
  void* dst2T = reinterpret_cast<void*>(out2T->getUnsafePtr());
  
  // Addresser<srcType> tOutput(dstT, scale[3], offset[3]);
  // const Addresser<srcType> tInput(srcT, scale[0], offset[0]);
  // Addresser<indexType> tRanges(prangesT, scale[1], offset[1]);
  // Addresser<indexType> tLengths(dst2T, scale[2], offset[2]);
  Addresser<srcType> tOutput(dstT, outT->dbggetscale(), outT->dbggetoffset());
  const Addresser<srcType> tInput(srcT, inT->dbggetscale(), inT->dbggetoffset());
  Addresser<indexType> tRanges(prangesT, rangesT->dbggetscale(), rangesT->dbggetoffset());
  Addresser<indexType> tLengths(dst2T, out2T->dbggetscale(), out2T->dbggetoffset());

  // unsigned int *srcIndex = (unsigned int *)srcDims;
  dim_t srcIndex[max_tensor_dimensions] = {0,};
  inT->dims(srcIndex);   
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  dim_t dstIndex[max_tensor_dimensions] = {0,};
  outT->dims(dstIndex);   
  // unsigned int *rangesIndex = (unsigned int *)prangesDims;
  dim_t rangesIndex[max_tensor_dimensions] = {0,};
  rangesT->dims(rangesIndex);    
  // unsigned int *lenIndex = (unsigned int *)dst2Dims;
  dim_t lenIndex[max_tensor_dimensions] = {0,};
  out2T->dims(lenIndex);

  // unsigned int *srcPitch = (unsigned int *)srcPitches;
  dim_t srcPitch[max_tensor_dimensions] = {0,};
  inT->dbgcpypitches(srcPitch);  
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  dim_t dstPitch[max_tensor_dimensions] = {0,};
  outT->dbgcpypitches(dstPitch);    
  // unsigned int *rangesPitch = (unsigned int *)prangesPitches;
  dim_t rangesPitch[max_tensor_dimensions] = {0,};
  rangesT->dbgcpypitches(rangesPitch);    
  // unsigned int *lenPitch = (unsigned int *)dst2Pitches;
  dim_t lenPitch[max_tensor_dimensions] = {0,};
  out2T->dbgcpypitches(lenPitch);

  unsigned int last_minion = activeMinions - 1;

  size_t typeSize = getsize<srcType>();
  unsigned int initialAddr = 0, maxRead = 0;

  if (minionId < last_minion) {

    unsigned int numElemsDst = dstPitch[0]*dstIndex[0];

    getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions - 1);
    if (maxRead == 0)
      return;

    // Assumption: srcDimsNum = dstDimsNum.
    unsigned int srcDimsNum = static_cast<unsigned int>(inT->dbggetnumdims());
    
    unsigned int coordOut[srcDimsNum];
    unsigned int last_non_zero_coord;
    getNonPaddingCoordinates(coordOut, initialAddr, srcDimsNum, dstPitch,
                             dstIndex, last_non_zero_coord);

    uint64_t offsetOut = 0;
    for (unsigned int i = 0; i < last_non_zero_coord; i++) {
      offsetOut += dstPitch[i]*coordOut[i];
    }

    uint64_t offsetRanges = rangesPitch[2];
    indexType length = tRanges[offsetRanges];
    unsigned int range = 0;
    unsigned int accumLength = 0;
    unsigned int exampleSize = rangesIndex[1];
    unsigned int exampleMem = rangesIndex[1]*rangesPitch[1];
    while ( static_cast<uint64_t> ((accumLength + length)*dstPitch[0]) < offsetOut) {
      accumLength += length;
      offsetRanges += rangesPitch[1];
      length = tRanges[offsetRanges];
      range++;
      if (range == exampleSize) {
        offsetRanges += rangesPitch[0] - exampleMem;
        range = 0;
      }
    }
    offsetRanges -= rangesPitch[2];

    uint64_t offsetIn = tRanges[offsetRanges]*srcPitch[0]; // tRanges[offsetRanges] is the starting batch id.
    indexType count = 1;
    while (static_cast<uint64_t> ((accumLength + count)*dstPitch[0]) < offsetOut) {
      offsetIn += srcPitch[0];
      count++;
    }
    count--;
    unsigned int positionInBatch = offsetOut - (accumLength + count)*dstPitch[0];
    offsetIn += positionInBatch;
    unsigned int coordIn[srcDimsNum];
    getNonPaddingCoordinates(coordIn, offsetIn, srcDimsNum, srcPitch, srcIndex,
                             last_non_zero_coord); // useless last parameter.

    unsigned int batchElems = 1;
    for (unsigned i = 1; i < srcDimsNum; ++i) batchElems *= srcIndex[i]; // avoiding padding elements.

    unsigned int posMax = maxRead + initialAddr;
    bool done = false;
    //TODO: SW-2650    bool doneIn = false; // useful for skipping padding positions in the source tensor.

    while (!done && (offsetOut < posMax)) {
      tOutput[offsetOut] = tInput[offsetIn];
      done = getOffsets(srcDimsNum, coordOut, offsetOut, dstIndex, dstPitch);
      positionInBatch++;
      if (positionInBatch != batchElems) {
        /*TODO: SW-2650 doneIn = */ getOffsets(srcDimsNum, coordIn, offsetIn, srcIndex, srcPitch);
      }
      else {
        positionInBatch = 0;
        count++;
        if (count != length) {
         /*TODO: SW-2650 doneIn = */getOffsets(srcDimsNum, coordIn, offsetIn, srcIndex, srcPitch);
        }
        else {
          count = 0;
          ++range;
          if (range != exampleSize) offsetRanges += rangesPitch[1];
          else {
            range = 0;
            offsetRanges += rangesPitch[0] - (exampleSize - 1)*rangesPitch[1];
          }
          offsetIn = tRanges[offsetRanges];
          length = tRanges[offsetRanges + rangesPitch[2]];
        }
      }
    }

    if (!DO_EVICTS) return;
    unsigned int clperminion = maxRead*sizeof(srcType)/CACHE_LINE_BYTES;
    if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
  }

// For coherence reasons only one minion should be able to write on the Length tensor, since in practice
// it will just be a short vector (not more than a couple cl's). This implementation involves the last
// active minion.

  else if (minionId == last_minion) {
    unsigned int numExamples = rangesIndex[0];
    unsigned int exampleSize = rangesIndex[1];
    unsigned int offsetRanges = rangesPitch[2];
    unsigned int auxoffsetRanges = rangesPitch[2]; // this aux variable helps avoiding products.
    unsigned int offsetLengths = 0;

    //It isn't thought that tlength writes will have more than 16 cache_lines.
    //So that, the initialAddr doesn't update.
    initialAddr = 0;

    for (size_t example = 0; example < numExamples; example++) { // size_t or indexType?
      indexType totalLength = 0;
      for (size_t range = 0; range < exampleSize; range++) {
          totalLength += tRanges[offsetRanges];
          offsetRanges += rangesPitch[1];
      }
      tLengths[offsetLengths] = totalLength;
      offsetRanges = auxoffsetRanges + rangesPitch[0];
      auxoffsetRanges = offsetRanges;
      offsetLengths += lenPitch[0];
    }

    // Todo: initialAddr should be the virtual address of the Length tensor.
    if (!DO_EVICTS) return;
    unsigned int clperminion = lenIndex[0]*lenPitch[0]*sizeof(srcType)/CACHE_LINE_BYTES;
    if (clperminion > 0) {
      evict_va_multi(DO_EVICTS, (uintptr_t)dst2T + typeSize*initialAddr, clperminion);
    }
    else {
      //evict all cache line even it wasn't completely written
      evict_va_multi(DO_EVICTS, (uintptr_t)dst2T + typeSize*initialAddr, 1);      
    }
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _GATHER_RANGES_INST_H_
