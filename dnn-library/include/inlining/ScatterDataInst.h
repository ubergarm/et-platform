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

#ifndef _SCATTER_DATA_INST_H_
#define _SCATTER_DATA_INST_H_

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

struct dataToCopyXSliceDim {

  unsigned int nCopy = 1;
  unsigned int advSlice = 1;
  unsigned int advDst = 1;
  unsigned int sliceAddr = 0;
  unsigned int dstAddr = 0;
  unsigned int jmpDstAddr = 0;
  unsigned int jmpSlcAddr = 0;
};

template <typename ptrT, unsigned int nDim>
class WriteSliceToDst {
  public:
  static INLINE_ATTR void cpyIt(const dataToCopyXSliceDim sliceSteps[max_tensor_dimensions],
				const ptrT* tSlices, ptrT* tOutput,
				unsigned int maxDim,
				unsigned int& sliceAddr, unsigned int& dstAddr) 
  {
    if (nDim >= maxDim) {
	WriteSliceToDst<ptrT, nDim-1>::cpyIt(sliceSteps, tSlices, tOutput, maxDim, sliceAddr, dstAddr);
    }
    else {
      for (unsigned int k=0; k < sliceSteps[nDim].nCopy; k++) {
      
	WriteSliceToDst<ptrT, nDim-1>::cpyIt(sliceSteps, tSlices, tOutput, maxDim, sliceAddr, dstAddr);
      
	sliceAddr += sliceSteps[nDim].sliceAddr;
	dstAddr += sliceSteps[nDim].dstAddr;
      }
    }
  }  
};

template <typename ptrT>
class WriteSliceToDst<ptrT, 0> {
  public:
  static INLINE_ATTR void cpyIt(const dataToCopyXSliceDim sliceSteps[max_tensor_dimensions],
				const ptrT* tSlices, ptrT* tOutput,
				unsigned int maxDim,
				unsigned int& sliceAddr, unsigned int& dstAddr) 
  {
    for (unsigned int k=0; k < sliceSteps[0].nCopy; k++) {
      auto val = tSlices[(sliceAddr + k)];
      tOutput[(dstAddr + k)] = val;
    }
  }
};

template <ElemKind elK>
inline void fwdLibScatterDataInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                                  uint64_t flags, const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using srcType = typename elemKind2elemTy<elK>::type;

  if (get_minion_id() != minionOffset) return;
  
  /* maintain compatibility through the new Iface Libtensor */
  /* outT --> dst  in1T--> src in2T--> slice*/
  
 
  // ptrType* tSlices = static_cast<ptrType*>(slicesT);
  srcType* tSlices = in2T->getRawDataPointer<srcType>();
  // ptrType* tOutput = static_cast<ptrType*>(dstT);
  srcType* tOutput = outT->getRawDataPointer<srcType>();
  //uint64_t  *tIndices = (uint64_t *)indexT;
  uint64_t* tIndices = in1T->getRawDataPointer<uint64_t>();
  
  // unsigned int *dstIndex = (unsigned int *)dstDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *indicesIndex = (unsigned int *)indicesDims;
  const dim_t *indicesIndex = in1T->dims().data();
  // unsigned int *slicesIndex = (unsigned int *)slicesDims;
  const dim_t *slicesIndex = in2T->dims().data();

  // unsigned int *dstPitch = (unsigned int *)dstPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *indicesPitch = (unsigned int *)pindicesPitches;
  const dim_t *indicesPitch = in1T->strides().data();
  // unsigned int *slicesPitch = (unsigned int *)slicesPitches;
  const dim_t * slicesPitch = in2T->strides().data();

  unsigned int sliceNumDim = static_cast<unsigned int>(in2T->ndims());
  
  dataToCopyXSliceDim sliceSteps[max_tensor_dimensions];
  int push_ptr = 0;

  //assert(sliceNumDim>1);
  unsigned int i=0;
  unsigned int accumSlc = 0;
  unsigned int accumDst = 0;
      
  for (i = 0; i < (sliceNumDim-1); i++) {

    sliceSteps[push_ptr++] = dataToCopyXSliceDim();    

    sliceSteps[i].nCopy = slicesIndex[sliceNumDim-(i+1)];
    
    accumSlc = 0;
    accumDst = 0;

    sliceSteps[i].advSlice = slicesPitch[(sliceNumDim-(i+1))];
    sliceSteps[i].advDst = dstPitch[(sliceNumDim-(i+1))];
    sliceSteps[i].jmpSlcAddr = sliceSteps[i].nCopy * sliceSteps[i].advSlice;
    sliceSteps[i].jmpDstAddr = sliceSteps[i].nCopy * sliceSteps[i].advDst;
    
    if (i == 1) {
      sliceSteps[i].sliceAddr = sliceSteps[i].advSlice;
      sliceSteps[i].dstAddr = sliceSteps[i].advDst;
    }
    else if (i>1) {

      for (unsigned int j = 1; j < i; j++) {
        accumSlc += sliceSteps[j].jmpSlcAddr;
        accumDst += sliceSteps[j].jmpDstAddr;
      }

      sliceSteps[i].sliceAddr = sliceSteps[i].advSlice - accumSlc;
      sliceSteps[i].dstAddr = sliceSteps[i].advDst - accumDst;
    }
  }

  for (unsigned int i = 0; i < indicesIndex[0]; i++) {
    unsigned int dstDataIdx = 0;
    for (unsigned int j = 0; j < indicesIndex[1]; j++) {
      dstDataIdx *= dstIndex[j];
      dstDataIdx += tIndices[(i*indicesPitch[0])+(j*indicesPitch[1])];
    }

    unsigned int dstAddr = (dstDataIdx * dstPitch[0]);
    unsigned int sliceAddr = (i * slicesPitch[0]);

    //  Non-recursive version keep it just in case of the 
    //  lower recursive perfomance 
    //
    // for (unsigned int k5 = 0; k5 < sliceSteps[5].nCopy; k5++) {
    //   for (unsigned int k4 = 0; k4 < sliceSteps[4].nCopy; k4++) {
    // 	for (unsigned int k3 = 0; k3 < sliceSteps[3].nCopy; k3++) {
    // 	  for (unsigned int k2 = 0; k2 < sliceSteps[2].nCopy; k2++) {
    // 	    for (unsigned int k1 = 0; k1 < sliceSteps[1].nCopy; k1++) {
    // 	      for (unsigned int k0 = 0; k0 < sliceSteps[0].nCopy; k0++) {
    // 		auto val = tSlices[(sliceAddr + k0)];
    // 		tOutput[(dstAddr + k0)] = val;
    // 	      }
    // 	      sliceAddr += sliceSteps[1].sliceAddr;
    // 	      dstAddr += sliceSteps[1].dstAddr;
    // 	    }
    // 	    sliceAddr += sliceSteps[2].sliceAddr;
    // 	    dstAddr += sliceSteps[2].dstAddr;
    // 	  }
    // 	  sliceAddr += sliceSteps[3].sliceAddr;
    // 	  dstAddr += sliceSteps[3].dstAddr;
    // 	}
    // 	sliceAddr += sliceSteps[4].sliceAddr;
    // 	dstAddr += sliceSteps[4].dstAddr;
    //   }
    //   sliceAddr += sliceSteps[5].sliceAddr;
    //   dstAddr += sliceSteps[5].dstAddr;
    // }

    WriteSliceToDst<srcType, (max_tensor_dimensions-1)>::cpyIt(sliceSteps, tSlices, tOutput, sliceNumDim-1, sliceAddr, dstAddr);
  }
}

 
// template <typename ElemTy>
// void BoundInterpreterFunction::fwdScatterDataInstCopyImpl() {
//   Tensor *dataT = getTensor(I->getData());
//   Tensor *indicesT = getTensor(I->getIndices());
//   Tensor *slicesT = getTensor(I->getSlices());

//   assert(indicesT->dims().size() == 2 &&
//          "Index should be stored in 2D tensor!");
//   const dim_t dataSliceSize = slicesT->size() / slicesT->dims()[0] *
//                               slicesT->getType().getElementSize();

//   auto IH = indicesT->getHandle<int64_t>();
//   // For each index, copy from the slice at that index into the location in data
//   // given the offset from the indices tensor.
//   for (dim_t i = 0, end = indicesT->dims()[0]; i < end; i++) {
//     dim_t destDataIdx = 0;
//     for (dim_t j = 0, e = indicesT->dims()[1]; j < e; j++) {
//       destDataIdx *= dataT->dims()[j];
//       destDataIdx += IH.at({i, j});
//     }
//     std::copy(&slicesT->getUnsafePtr()[i * dataSliceSize],
//               &slicesT->getUnsafePtr()[(i + 1) * dataSliceSize],
//               &dataT->getUnsafePtr()[dataSliceSize * destDataIdx]);
//   }
// } 

} // namespace inlining

} // namespace dnn_lib

#endif // _SCATTER_DATA_INST_H_
