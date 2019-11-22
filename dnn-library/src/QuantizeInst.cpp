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

/// Quantize floating point tensor. Scale and Offset are based on return type
/// of the instruction \p I.
template <typename dstType>
void dnn_lib::fwdLibQuantizeInst(void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, float scale,
                                 int32_t offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<dstType> ptrDstT(dstT, scale, offset);
  float *ptrSrcT = (float *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int eDims[MAX_TENSOR_DIMENSIONS] = {1, 1, 1, 1, 1, 1};
  unsigned int eDstPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};
  unsigned int eSrcPitch[MAX_TENSOR_DIMENSIONS] = {0, 0, 0, 0, 0, 0};

  for (size_t i = 0; i < srcDimNum; i++) {
    eDims[i] = srcIndex[i];
    eDstPitch[i] = dstPitch[i];
    eSrcPitch[i] = srcPitch[i];
  }

  for (size_t x = 0; x < eDims[0]; x++) {
    for (size_t y = 0; y < eDims[1]; y++) {
      for (size_t z = 0; z < eDims[2]; z++) {
        for (size_t w = 0; w < eDims[3]; w++) {
          for (size_t q = 0; q < eDims[4]; q++) {
            for (size_t r = 0; r < eDims[5]; r++) {
              int64_t dstAddr = x * eDstPitch[0] + y * eDstPitch[1] +
                                z * eDstPitch[2] + w * eDstPitch[3] +
                                q * eDstPitch[4] + r * eDstPitch[5];
              int64_t srcAddr = x * eSrcPitch[0] + y * eSrcPitch[1] +
                                z * eSrcPitch[2] + w * eSrcPitch[3] +
                                q * eSrcPitch[4] + r * eDstPitch[5];
              auto val = ptrSrcT[srcAddr];
              // TODO check if we can use Addresser without breaking all the
              // other tests that uses int32_t as non quantized type
              if (std::is_same<dstType, int32_t>::value) {
                ptrDstT[dstAddr] = quantize<int32_t>(val, scale, offset);
              } else {
                ptrDstT[dstAddr] = val;
              }
            }
          }
        }
      }
    }
  }
}
template <typename dstType>
void dnn_lib::fwdLibQuantizeInstThreaded(void *dstT, void *dstDims,
                                         void *dstPitches, void *srcT,
                                         void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float scale,
                                         int32_t offset, uint64_t flags) {
  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<dstType> ptrDstT(dstT, scale, offset);
  float *ptrSrcT = (float *)srcT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *srcIndex = (unsigned int *)srcDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *srcPitch = (unsigned int *)srcPitches;

  unsigned int numElemsDst =
      dstPitch[0] * srcIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  // We move the initialAddr to the next non-padding position
  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k = 0;              // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, srcIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetIn = 0;
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetIn += srcPitch[j] * coord[j];
    offsetOut += dstPitch[j] * coord[j];
  }

  unsigned int posMax = maxRead + initialAddr;
  // In each iteration we copy a position and switch to the next one, until
  // completion.
  bool done = false;
  while (!done) {
    if (offsetOut >= posMax)
      break;
    auto val = ptrSrcT[offsetIn];
    // TODO check if we can use Addresser without breaking all the
    // other tests that uses int32_t as non quantized type
    if (std::is_same<dstType, int32_t>::value) {
      ptrDstT[offsetOut] = quantize<int32_t>(val, scale, offset);
    } else {
      ptrDstT[offsetOut] = val;
    }
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, srcIndex,
                      srcPitch, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}
