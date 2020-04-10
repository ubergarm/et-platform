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

namespace dnn_lib {

namespace inlining {

template <typename srcType>
void fwdLibSplatInst(void *dst, void *dstDims,
                     void *dstPitches, unsigned int dstDimNum,
                     uint64_t *splatVal, const float *scale,
                     const int32_t *offset, uint64_t flags) {
  
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  size_t numElems = dstIndex[0] * dstPitch[0];
  
  uint64_t *dst64 = static_cast<uint64_t*>(dst);
  // splatVal has the data replicated as many times as to fill a uint64 
  constexpr size_t ratio64 = sizeof(uint64_t) / sizeof(srcType);
  constexpr size_t mask = ratio64 - 1;
  static_assert( (ratio64 & (ratio64 - 1)) == 0, "ratio to 64b word is not power of 2" );


  for (size_t i = 0 ; i < (static_cast<size_t>(numElems) & (~mask)); i++, dst64++) 
    *dst64 = *splatVal;

  memcpy(dst64, splatVal, (numElems & mask) * sizeof(srcType));

}

template <typename sourceTy>
inline void fwdLibSplatInstThreaded(void *dst, void *dstDims,
                                      void *dstPitches, unsigned int dstDimNum,
                                      uint64_t *splatValPtr, const float *scale,
                                      const int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  using srcType = typename std::conditional< std::is_same<sourceTy, float16>::value, uint16_t, sourceTy>::type;

  srcType *tOutput = (srcType *)dst;
  srcType splatVal = bitwise_lsb_copy<srcType> (*splatValPtr);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;

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
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <typename srcType>
inline void fwdLibSplatInstVectorized(void *dst, void *dstDims,
                                        void *dstPitches, unsigned int dstDimNum,
                                        uint64_t *splatVal, const float *scale,
                                        const int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  size_t typeSize = getsize<srcType>();
  size_t bytesperCL = 64;

  uint64_t totalBytes = dstPitch[0] * dstIndex[0] * typeSize;
  uint64_t totalCL = (totalBytes - 1)/bytesperCL + 1;
  uint64_t CLperMinion = (totalCL - 1)/activeMinions + 1;
  uint64_t startCL = minionId * CLperMinion;

  if (startCL >= totalCL) return;
  if (startCL + CLperMinion > totalCL) CLperMinion = totalCL - startCL;

  uint64_t regsperMinion = 2 * CLperMinion;  // A cacheline contains 2 regs
  uint64_t offsetOut = startCL * 64;

  char *dstPtr = (char *)dst;
  dstPtr += offsetOut;

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
         [splatValPtr] "r" (splatVal)

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
