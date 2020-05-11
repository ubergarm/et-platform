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

#ifndef _CONVERT_TO_INST_H_
#define _CONVERT_TO_INST_H_

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


// if defined, it is OK to write in padding (faster if we can ignore setting masks)
#define CONVERTTO_OK_TO_WRITE_PADDING


namespace dnn_lib {

namespace inlining {

template <typename srcType, typename dstType>
inline __attribute__((always_inline)) void fwdLibConvertToInst(LibTensor* inT, LibTensor* outT) {
  // FIXME: single thread convertto fails when combined with multi-threaded
  // operators
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  assert(inT->dims() == outT->dims());
  

  /* maintain compatibility through the new Iface Libtensor */

  void* srcT = inT->getRawDataPointer<void>();
  void* dstT = outT->getRawDataPointer<void>();

  // Addresser<dstType> ptrDstT(dstT, scalexo[1], offset[1]);
  Addresser<dstType> ptrDstT(dstT, outT->getScale(), outT->getOffset());
  // const Addresser<srcType> ptrSrcT1(srcT1, scale[0], offset[0]);
  const Addresser<srcType> ptrSrcT1(srcT, inT->getScale(), inT->getOffset());

  Converter<srcType, dstType> converter;

  dims_loop<>::run(outT->dims(), outT->strides(), inT->strides(),
                   [&](size_t addrDst, size_t addrSrc) {
                     auto src = ptrSrcT1[addrSrc];
                     if (std::is_same<srcType, dstType>::value) {
                       ptrDstT[addrDst] = src;
                     } else {
                       auto dst = converter.convert(src);
                       ptrDstT[addrDst] = dst;
                     }
                   } );

}

template <typename srcType, typename dstType>
inline __attribute__((always_inline)) void fwdLibConvertToInstThreaded(LibTensor* inT, LibTensor* outT, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  size_t activeMinions =  MIN_PER_SHIRE * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  ////////////////////////////////////////////////////////////////////////////////
  // partition work between minions in multiples of CL
  ////////////////////////////////////////////////////////////////////////////////  
  size_t first; // first element in raw array to process
  size_t count; // nr  elements in to process (will be in multiples of CL)

  outT->partitionCL(minionId, activeMinions, first, count);

  if (unlikely(count == 0)) return; // minion has no work to do
  
  /* maintain compatibility through the new Iface Libtensor */

  void* src = inT->getRawDataPointer<void>();
  void* dst = outT->getRawDataPointer<void>();

  Addresser<dstType> tOutput(dst, outT->getScale(), outT->getOffset());
  const Addresser<srcType> tAInput(src, inT->getScale(), inT->getOffset());
  Converter<srcType, dstType> converter;
  
  // and loop
#if 0
  auto out = outT->getHandle<dstType>().getIterator(first);
  auto in = inT->getHandle<srcType>().getIterator(out.coords());
  
  for(; out.offset() < first + count;++out, ++in) {
    auto inD = tAInput[in.offset()];
    auto outD = converter.convert(inD);
    tOutput[out.offset()] = outD;
  }
#else

  auto begin = outT->offset2Coord(first);
  //  auto end = outT->offset2Coord(first + count);
  dims_loop<>::run(outT->dims(), outT->strides(), inT->strides(),
                   begin, first + count, 
                   [&](size_t addrDst, size_t addrSrc) {
                     auto inD = tAInput[addrSrc];
                     auto outD = converter.convert(inD);
                     tOutput[addrDst] = outD;
                   } );
#endif
  outT->evict(DO_EVICTS, first, count);
  
}

  
template <typename srcType, typename dstType>
inline __attribute__((always_inline)) void fwdLibConvertToInstVectorized(LibTensor* inT,  LibTensor*outT, uint64_t flags){
  const unsigned int minionId = get_minion_id();
  const unsigned int activeMinions = MIN_PER_SHIRE * ACTIVE_SHIRES;
  assume(activeMinions<=1024);
  if (minionId >= activeMinions)
    return;

  ////////////////////////////////////////////////////////////////////////////////
  // partition work between minions in multiples of CL
  ////////////////////////////////////////////////////////////////////////////////  
  dim_t first; // first element in raw array to process
  dim_t count; // nr  elements in to process (will be in multiples of CL)

  outT->partitionCL(minionId, activeMinions, first, count);
  if (unlikely(count == 0)) return; // minion has no work to do

  Converter<srcType, dstType> converter;
  int32_t gatherValues[8], scatterValues[8];
  for (unsigned int i = 0; i < 8; i++) {
    gatherValues[i] = i * getsize<srcType>();
    scatterValues[i] = i * getsize<dstType>();
  }
  // and loop

  //  srcType* srcP = inT->getRawDataPointer<srcType>();
  //  dstType* dstP = outT->getRawDataPointer<dstType>();
  //FIXME:  dstType=float16? cannot use as pointers to storage! should be uint16_, this is just a temporary fix
  
  using srcStorage_t = typename std::conditional<std::is_same<srcType, float16>::value,
                                                 uint16_t, srcType>::type;
  using dstStorage_t = typename std::conditional<std::is_same<dstType, float16>::value,
                                                 uint16_t, dstType>::type;

  srcStorage_t * const  srcP = inT->getRawDataPointer<srcStorage_t>();
  dstStorage_t * const  dstP = outT->getRawDataPointer<dstStorage_t>();
  
  const size_t ndims = outT->ndims();
  const dim_t lastDim = outT->dims()[ndims-1];
  constexpr dim_t step=8; // ConvertVect works with 8 elements at a time

  // get iterators to loop through all the dimensions except the last one
  auto out = outT->getHandle<dstStorage_t>().getIterator(first);
  auto in = inT->getHandle<srcStorage_t>().getIterator(out.coords());
  
#if 0
  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n"); // set initial mask
  for(; out.offset() < first + count; out+=step, in+=step) {
    dim_t valid = lastDim - out.coords()[ndims-1];
    
    // set and restore the mask if we are in the boundary before and after the conversion
    if ( valid < step) __asm__ __volatile__ ("mov.m.x m0, %0, 0" : : "r" ((1ULL << valid) -1 ));
    
    //conversion
    converter.convertVect( reinterpret_cast<uintptr_t>(srcP + in.offset() ),
                           reinterpret_cast<uintptr_t>(dstP + out.offset() ),
                           gatherValues, scatterValues);
    // and restore mask
    if ( valid < step)  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n"); // set initial mask
    
  }
#else
  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n"); // set initial mask
  dim_t endOffset = first + count;
  dim_t oOffset = out.offset();
  dim_t iOffset = in.offset();
  
  ////////////////////////////////////////////////////////////////////////////////
  // simpler loop if there is just one dimension
  ////////////////////////////////////////////////////////////////////////////////
  if (ndims == 1) {
    for( ; oOffset < endOffset; oOffset+=step, iOffset+=step){
#ifndef  CONVERTTO_OK_TO_WRITE_PADDING
      dim_t valid = lastDim - out.coords()[ndims-1];
      // set and restore the mask if we are in the boundary before and after the conversion
      if ( valid < step) __asm__ __volatile__ ("mov.m.x m0, %0, 0" : : "r" ((1ULL << valid) -1 ));
#endif
      //conversion
      converter.convertVect( reinterpret_cast<uintptr_t>(srcP + iOffset),
                             reinterpret_cast<uintptr_t>(dstP + oOffset),
                             gatherValues, scatterValues);
      
    }
    
  }
  else {
    ////////////////////////////////////////////////////////////////////////////////
    // Loops for more than 1 dim
    ////////////////////////////////////////////////////////////////////////////////
    
    //// First iterate until completing the first feature dimension (in case initial coordinates are in the middle)
    for(; ( out.coords()[ndims-1] != 0 ||  ndims == 1) && out.offset() < endOffset; out+=step, in+=step) {
    dim_t valid = lastDim - out.coords()[ndims-1];
#ifndef  CONVERTTO_OK_TO_WRITE_PADDING
    // set and restore the mask if we are in the boundary before and after the conversion
    if ( valid < step) __asm__ __volatile__ ("mov.m.x m0, %0, 0" : : "r" ((1ULL << valid) -1 ));
#endif
    
    //conversion
    converter.convertVect( reinterpret_cast<uintptr_t>(srcP + iOffset),
                           reinterpret_cast<uintptr_t>(dstP + oOffset),
                           gatherValues, scatterValues);
    iOffset+=step; oOffset+=step;
    }
    
    //// Then, complete the remaining iterators
    __asm__ __volatile__("mov.m.x m0, zero, 0xff \n"); // set initial mask
    for( ;out.offset() < endOffset ; out.step(ndims-2), in.step(ndims-2) ){ // step 2n outer dimension
      assume(out.coords()[ndims-1] == 0 && in.coords()[ndims-1] == 0);
      for ( dim_t i = 0 ; i < lastDim && oOffset + i < endOffset; i+=step) { // step outer dimension
#ifndef  CONVERTTO_OK_TO_WRITE_PADDING        
        dim_t valid = lastDim - i;
        // set and restore the mask if we are in the boundary before and after the conversion
        if ( valid < step) __asm__ __volatile__ ("mov.m.x m0, %0, 0" : : "r" ((1ULL << valid) -1 ));
#endif
        
        //conversion
        converter.convertVect( reinterpret_cast<uintptr_t>(srcP + in.offset()  + i),
                               reinterpret_cast<uintptr_t>(dstP + out.offset() + i ),
                               gatherValues, scatterValues);
#ifndef CONVERTTO_OK_TO_WRITE_PADDING        
        // and restore mask
        if ( valid < step)  __asm__ __volatile__("mov.m.x m0, zero, 0xff \n"); // set initial mask
#endif
      }
    }
  }
  
#endif


  
  // and evict if need be
  outT->evict(DO_EVICTS, first, count);
  
}
  
} // namespace inlining

} // namespace dnn_lib

#endif // _CONVERT_TO_INST_H_
