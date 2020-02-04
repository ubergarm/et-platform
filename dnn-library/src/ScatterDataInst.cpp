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

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>
#include "LibNodes.h"
#include "GenInstances.h"
#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"


using namespace std;

#define MAX_DIM_ALLOWED 6

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
  static INLINE_ATTR void cpyIt(const dataToCopyXSliceDim sliceSteps[MAX_DIM_ALLOWED],
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
  static INLINE_ATTR void cpyIt(const dataToCopyXSliceDim sliceSteps[MAX_DIM_ALLOWED],
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

template <typename srcType>
void dnn_lib::fwdLibScatterDataInst(void *dstT, void *dstDims,
				    void *dstPitches, unsigned int dstNumDim, void *indexT,
				    void *indicesDims, void *pindicesPitches,
				    void *slicesT, void *slicesDims, unsigned int sliceSize,
				    void *slicesPitches, unsigned int sliceNumDim, float *scale,
				    int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  using ptrType = UintSelector<getsize<srcType>()>;

  ptrType* tSlices = static_cast<ptrType*>(slicesT);
  ptrType* tOutput = static_cast<ptrType*>(dstT);
  
  uint64_t  *tIndices = (uint64_t *)indexT;
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *indicesIndex = (unsigned int *)indicesDims;
  unsigned int *slicesIndex = (unsigned int *)slicesDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *indicesPitch = (unsigned int *)pindicesPitches;
  unsigned int *slicesPitch = (unsigned int *)slicesPitches;

  dataToCopyXSliceDim sliceSteps[MAX_DIM_ALLOWED];
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

    WriteSliceToDst<ptrType, (MAX_DIM_ALLOWED-1)>::cpyIt(sliceSteps, tSlices, tOutput, sliceNumDim-1, sliceAddr, dstAddr);
  }
}


template <typename srcType>
void dnn_lib::fwdLibScatterDataInstThreaded(void *dstT, void *dstDims,
                                              void *dstPitches, unsigned int dstDimNum, void *indexT,
                                              void *indicesDims, void *pindicesPitches,
                                              void *slicesT, void *slicesDims,
                                              void *slicesPitches, float *scale,
                                              int32_t *offset, uint64_t flags) {


  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  const Addresser<srcType> tSlices(slicesT, scale[0], offset[0]);
  Addresser<srcType> tOutput(dstT, scale[2], offset[2]);

  long long *tIndices = (long long *)indexT;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *indicesIndex = (unsigned int *)indicesDims;
  unsigned int *slicesIndex = (unsigned int *)slicesDims;

  unsigned int *dstPitch = (unsigned int *)dstPitches;
  unsigned int *indicesPitch = (unsigned int *)pindicesPitches;
  unsigned int *slicesPitch = (unsigned int *)slicesPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[dstDimNum];
  for (unsigned int i = 0; i < dstDimNum; i++)
    coord[i] = 0;
  unsigned int k;
  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex,
                           k);
  
  unsigned int offsetIn = 0; // Doesn't include slicesPitch[0]. offsetIn doesn't
                             // have the conventional meaning
  unsigned int offsetOut = dstPitch[0] * coord[0];

  unsigned int slicesPitch_0 = slicesPitch[0];
  slicesPitch[0] = 0;

  
  for (unsigned int j = 1; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
    offsetIn += slicesPitch[j] * coord[j];
  }
 
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  bool change = true;
  int offset0 = 0;

  while (!done && (offsetOut < posMax)) {
    if (change){
      offset0 = indicesIndex[0] - 1;
      while(offset0 >= 0){
        if (tIndices[offset0] == coord[0]) break;
        offset0--;
      }
      change = false;
    }

    if (offset0 >= 0) tOutput[offsetOut] = tSlices[offsetIn + offset0*slicesPitch_0];

    for (int j = dstDimNum - 1; j >= 0; j--) {
      if (coord[j] != (dstIndex[j] - 1)) {
        offsetOut += dstPitch[j];
        offsetIn += slicesPitch[j];
        coord[j]++;
        if (j == 0) change = true;
        break;
      } else if (j != 0) {
        offsetOut -= (dstIndex[j] - 1) * dstPitch[j];
        offsetIn -= (slicesIndex[j] - 1) * slicesPitch[j];
        coord[j] = 0;
      } else {
        done = true;
        break;
        }
    }
  }

  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstT + typeSize*initialAddr, clperminion);
}

GEN_INSTANCES_OP(template, fwdLibScatterDataInst, void *dstT, void *dstDims,
		 void *dstPitches, unsigned int dstNumDim, void *indexT, void *indicesDims, void *pindicesPitches,
		 void *slicesT, void *slicesDims, unsigned int sliceSize, void *slicesPitches, unsigned int sliceNumDim,
		 float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibScatterDataInstThreaded, void *dstT, void *dstDims,
		 void *dstPitches, unsigned int dstDimNum, void *indexT, void *indicesDims, void *pindicesPitches,
		 void *slicesT, void *slicesDims, void *slicesPitches ,
		 float *scale, int32_t *offset, uint64_t flags);
