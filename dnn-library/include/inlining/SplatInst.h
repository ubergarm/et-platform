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

#ifndef _SPLAT_INST_H_
#define _SPLAT_INST_H_

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

template <ElemKind elK>
inline uint64_t extendSplatValue64(const LibTensor *outT, const float splatVal){
  using srcType = typename elemKind2elemTy<elK>::type;
  // convert splatVal to srcType
  srcType splatValType;
  Addresser<elK> valAd(&splatValType, outT->getScale(), outT->getOffset());
  valAd[0] = splatVal;
  //  and replicate to fit up to 64 bits
  uint64_t splatVal64 = bitwise_lsb_copy<uint64_t> (splatValType);
  for( size_t i = 1, j = 1; i < sizeof(uint64_t) / sizeof(srcType); i<<=1, j++){
    splatVal64 = splatVal64  | splatVal64 << (j*sizeof(srcType)*8);
  }
  return splatVal64;
}
  
template <ElemKind elK>
void fwdLibSplatInst(LibTensor *outT, const float splatVal,
                     uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  if (get_minion_id() != minionOffset) return;

  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();

  size_t numElems = dstIndex[0] * dstPitch[0];

  // transform splatVal to srcType, and replicate to fit up to 64 bits
  uint64_t splatVal64 = extendSplatValue64<elK>(outT, splatVal);
  
  uint64_t *dst64 = static_cast<uint64_t*>(dst);
  // splatVal has the data replicated as many times as to fill a uint64 
  constexpr size_t ratio64 = sizeof(uint64_t) / sizeof(srcType);
  constexpr size_t mask = ratio64 - 1;
  static_assert( (ratio64 & (ratio64 - 1)) == 0, "ratio to 64b word is not power of 2" );


  for (size_t i = 0 ; i < (static_cast<size_t>(numElems) & (~mask)); i+=ratio64, dst64++) {
    *dst64 = splatVal64;
  }

  memcpy(dst64, &splatVal64, (numElems & mask) * sizeof(srcType));

}

template <ElemKind elK>
inline void fwdLibSplatInstThreaded(LibTensor* outT, const float splatVal,
                                    uint64_t flags,
                                    const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  void *dst = outT->getRawDataPointer<void>();
  
  // srcType *tOutput = (srcType *)dst;
  srcType *tOutput = outT->getRawDataPointer<srcType>();
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();

  unsigned int dstDimNum = static_cast<unsigned int>(outT->ndims());

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  // Get minion id
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;
  unsigned int coord[dstDimNum]; // Vector of coordinates
  unsigned int k;                  // Amount of non-zero coordinates

  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    tOutput[offsetOut] = splatVal;
    done = getOffsets(dstDimNum, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <ElemKind elK>
inline void fwdLibSplatInstVectorized(LibTensor* outT, const float splatVal, uint64_t flags,
                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  void* dst = outT->getRawDataPointer<void>();

  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();

  
  size_t typeSize = getsize<srcType>();
  size_t bytesperCL = CACHE_LINE_BYTES;

  uint64_t totalBytes = dstPitch[0] * dstIndex[0] * typeSize;
  uint64_t totalCL = (totalBytes - 1)/bytesperCL + 1;
  uint64_t CLperMinion = (totalCL - 1)/activeMinions + 1;
  uint64_t startCL = minionId * CLperMinion;

  if (startCL >= totalCL) return;
  if (startCL + CLperMinion > totalCL) CLperMinion = totalCL - startCL;

  uint64_t regsperMinion = 2 * CLperMinion;  // A cacheline contains 2 regs
  uint64_t offsetOut = startCL * CACHE_LINE_BYTES;

  char *dstPtr = (char *)dst;
  dstPtr += offsetOut;

  // transform splatVal to srcType, and replicate to fit up to 64 bits
  uint64_t splatVal64 = extendSplatValue64<elK>(outT, splatVal);
  
  
  if (regsperMinion > 0){
    //TODO: it writes in padding! check if that's alright!
    
    // assuming dstPtr is aligned to 32b
    float scratch;
    char *endPtrUnrolled = dstPtr + (regsperMinion & (~(0x20*8 -1)) ) * 0x20; // unrolling 8 stores in the same iteration
    char *endPtr = dstPtr + regsperMinion * 0x20;
    __asm__ __volatile__
      (
       "mov.m.x m0, zero, 0xff\n"
       // replicate splatVal into the 8 lanes
       // splatVal has the value replicated in 2 lanes (uint64_t is the maxType)
       "fg32w.ps %[scratch], %[fg32w_conf](%[splatValPtr])\n"
       
#ifndef DNN_LIB_DO_NOT_UNROLL_LOOPS
       "beq %[endPtrUnrolled], %[dstPtr], 2f\n"
       "1:\n" // loop unrolling
       "fsw.ps %[scratch], 0x20*0(%[dstPtr])\n"
       "fsw.ps %[scratch], 0x20*1(%[dstPtr])\n"
       "fsw.ps %[scratch], 0x20*2(%[dstPtr])\n"
       "fsw.ps %[scratch], 0x20*3(%[dstPtr])\n"
       "fsw.ps %[scratch], 0x20*4(%[dstPtr])\n"
       "fsw.ps %[scratch], 0x20*5(%[dstPtr])\n"
       "fsw.ps %[scratch], 0x20*6(%[dstPtr])\n"
       "fsw.ps %[scratch], 0x20*7(%[dstPtr])\n"
       "addi %[dstPtr], %[dstPtr], 0x20*8\n"
       "bne %[endPtrUnrolled], %[dstPtr], 1b\n"
       
       "2: beq %[endPtr], %[dstPtr], 2f\n"
#endif     
       "1:\n" // last iterations (0..7)
       "fsw.ps %[scratch], 0x0(%[dstPtr])\n"
       "addi %[dstPtr], %[dstPtr], 0x20\n"
       "bne %[endPtr], %[dstPtr], 1b\n"
       "2:"
       
       : [dstPtr] "+&r" (dstPtr),      // pointer in xreg
         [scratch] "=&f" (scratch)  // scratch floating point register
         
       : [fg32w_conf] "r" (0x208208),   // gather32 configuration in xreg
         [endPtr] "r" (endPtr),
         [endPtrUnrolled] "r" (endPtrUnrolled),
         [splatValPtr] "r" (&splatVal64),
         [splatValPtrMem] "m" (splatVal64)

       : "memory"
     );
  }

  if (!DO_EVICTS)
    return;

  if (CLperMinion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + offsetOut, CLperMinion);

}

} // namespace inlining

} // namespace dnn_lib

#endif // _SPLAT_INST_H_
