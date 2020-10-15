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

#ifndef _FULLY_CONNECTED_INST_H_
#define _FULLY_CONNECTED_INST_H_

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

asm(".include \"Fully-macro-defs.S\"");

namespace dnn_lib {

namespace inlining {

// generic version is vectorized
template <ElemKind dstElK, ElemKind biasElK>
inline void fwdLibFullyConnectedInst(LibTensor* outT, LibTensor* in1T,
				     LibTensor* in2T, LibTensor* in3T,
				     uint64_t flags,
				     const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  using dstType  = typename elemKind2elemTy<dstElK>::type;
  using biasType = typename elemKind2elemTy<biasElK>::type;

  assert((dstElK == FloatTy) || (dstElK == Float16Ty) || (dstElK == Int8QTy) || 
	 (dstElK == Int16QTy));
  
  if (!isQuantizedElemKind(dstElK)) {
    assert(dstElK == biasElK);
  }
  else {
    if (dstElK == Int8QTy) {
      assert((biasElK == Int8QTy) || (biasElK == Int32QTy) || (biasElK == FloatTy));
    }
    else if(dstElK == Int16QTy) {
      assert((biasElK == Int16QTy) || (biasElK == Int32QTy));
    }
    else {
      assert(false && "Type unsupported");
    }
  }

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  float scale[] = { in1T->getScale(), in2T->getScale(), in3T->getScale(), outT->getScale()};
  int32_t offset[] = { in1T->getOffset(), in2T->getOffset(), in3T->getOffset(), outT->getOffset()};

  /* maintain compatibility through the new Iface Libtensor */
  /* outT --> dst  in1T--> inActT  in2T--> inWeighT in3T-->inBiasT */
  void *dstMatrix = outT->getRawDataPointer<void>();
  void *activations = in1T->getRawDataPointer<void>();
  void *weights = in2T->getRawDataPointer<void>();
  void *biases = in3T->getRawDataPointer<void>();

  const dim_t *dstIndex = outT->dims().data();
  const dim_t *actIndex = in1T->dims().data();
  
  const dim_t *dstPitch = outT->strides().data();
  const dim_t *actPitch = in1T->strides().data();
  const dim_t *weightPitch = in2T->strides().data();
  
  // Total number of elements to process is the size of the outter
  // dimension of the destination tensor multiplied by its pitch
  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = sizeof(dstType);
  size_t biasSize = sizeof(biasType);
  
  // Gets the total number of elements to work on for the minion
  // initialAddr: is first element to start working on
  // maxRead: number of elements to process
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions, dstMatrix);
  if (maxRead == 0)
    return;

  unsigned int coord[2];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 2, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  unsigned int offsetAIn = coord[0]*actPitch[0];

  int32_t gatherValuesWgt[8];
  gatherValuesWgt[0] = 0;
  unsigned int step = weightPitch[0]*typeSize;
  for (unsigned int i = 1; i < 8; ++i) {
    gatherValuesWgt[i] = gatherValuesWgt[i - 1] + step;
  }
  unsigned int wgtRegStep = 8*step;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    uintptr_t dstAddr = (uintptr_t)dstMatrix + typeSize*offsetOut;
    uintptr_t actAddr = (uintptr_t)activations + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)weights + typeSize*coord[1];
    uintptr_t biasAddr = (uintptr_t)biases + biasSize*coord[1];

    __asm__ __volatile__(

      "INIT_STEP %[gthValuesWgt], %[wgtRegStep], %[elemsRow], %[elk], %[belk]\n"
      "ACT_x_WGT %[actAddr], %[wgtAddr], %[scale], %[offset], %[elk]\n"
      "NEXT_CHUNK_STEP %[actAddr], %[elemsRow], %[elk]\n"
      "ACT_x_WGT %[actAddr], %[wgtAddr], %[scale], %[offset], %[elk]\n"
      "ADD_BIAS_STEP %[biasAddr], %[dstAddr], %[scale], %[offset], %[elk], %[belk]\n"
      
      : 
      : [gthValuesWgt] "m" (gatherValuesWgt),  
	[wgtRegStep] "r" (&wgtRegStep),
	[elemsRow] "r" (actIndex[1]),
	[actAddr] "r" (actAddr),
	[wgtAddr] "r" (wgtAddr),
	[dstAddr] "r" (dstAddr),
	[biasAddr] "r" (biasAddr),
	[scale] "r" (scale),                  //nullptr value to uniform parameters in asm macro call.
	[offset] "r" (offset),                //nullptr value to uniform parameters in asm macro call.
	[elk] "i" (dstElK),
	[belk] "i" (biasElK)
      : "t0", "t1", "f0", "f1", "f28", "f29", "f30", "f31", "memory");
   
    done = getOffsets(2, coord, offsetOut, dstIndex, dstPitch);
    if (coord[1] == 0) {
      offsetAIn += actPitch[0];
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _FULLY_CONNECTED_INST_H_
