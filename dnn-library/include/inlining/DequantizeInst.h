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

#ifndef _DEQUANTIZE_INST_H_
#define _DEQUANTIZE_INST_H_

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

/// Dequantize integer tensor. Scale and Offset are based
/// on the source tensor type.
template <ElemKind dstElK, ElemKind srcElK>
inline void fwdLibDequantizeInst(LibTensor* outT, LibTensor* inT, uint64_t flags,
                                 const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  using dstType = typename elemKind2elemTy<dstElK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;

  if (minionId >= activeMinions) {
    return;
  }

  /* maintain compatibility through the new Iface Libtensor */
  void* srcT = inT->getRawDataPointer<void>();
  void* dstT = outT->getRawDataPointer<void>();
 
  Addresser<dstElK> ptrDstT(dstT, outT->getScale(), outT->getOffset());
  const Addresser<srcElK> ptrSrcT(srcT, inT->getScale(), inT->getOffset());
  
  const dim_t *srcIndex = inT->dims().data();
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *srcPitch = inT->strides().data();

  unsigned int srcDimNum = static_cast<unsigned int>(inT->ndims());
  
  unsigned int numElemsDst =
      dstPitch[0] * srcIndex[0]; // Total number of elements in the tensor

  // We give to each minion an initial address the number of positions that it
  // must work on (maxRead).
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dstT);

  if (maxRead == 0) {
    return;
  }

  // We move the initialAddr to the next non-padding position
  unsigned int coord[srcDimNum]; // Vector of coordinates
  unsigned int k = 0;            // Amount of non-zero coordinates
  getNonPaddingCoordinates(coord, initialAddr, srcDimNum, dstPitch, srcIndex, k);

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
  while (not done and (offsetOut < posMax)) {
    ptrDstT[offsetOut] = ptrSrcT[offsetIn];
    done = getOffsets(srcDimNum, coord, offsetIn, offsetOut, srcIndex, srcPitch, dstPitch) or (offsetOut >= posMax);
  }

  if (!DO_EVICTS) {
    return;
  }
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) {
    evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize * initialAddr, clperminion);
  }
}

} // namespace inlining

} // namespace dnn_lib

#endif // _DEQUANTIZE_INST_H_
