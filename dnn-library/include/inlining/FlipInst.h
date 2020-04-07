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

#ifndef FLIP_INST_H_
#define FLIP_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "LibNodes.h"
#include "LibCommon.h"
#include "LibTensor.h"
#include "GenInstances.h"
#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"

namespace dnn_lib {

namespace inlining {


inline void expandDimsToMax(dim_t* newDims, dim_t* currDims, unsigned int numDims) {

  for (unsigned int i = 0; i< max_tensor_dimensions; i++) {
    if (i< numDims)
      newDims[i] = currDims[i];
    else
      newDims[i] = 1;
  }
}

template <typename ElemTy>
inline void loopAxis(Handle<ElemTy> srcH, Handle<ElemTy>  destH, dim_t *newDims, unsigned int axis) {

  dim_t indicesDest[max_tensor_dimensions] = {0,};
  dim_t auxDst[max_tensor_dimensions] = {0,};
  dim_t indicesSrc[max_tensor_dimensions] = {0,};
  dim_t ndx[max_tensor_dimensions] = {0,};
  
  
  for (ndx[0] = 0; ndx[0] < newDims[0]; ndx[0]++)
    for (ndx[1] = 0; ndx[1] < newDims[1]; ndx[1]++)
      for (ndx[2] = 0; ndx[2] < newDims[2]; ndx[2]++)
        for (ndx[3] = 0; ndx[3] < newDims[3]; ndx[3]++)
          for (ndx[4] = 0; ndx[4] < newDims[4]; ndx[4]++)
            for (ndx[5] = 0; ndx[5] < newDims[5]; ndx[5]++) {

              for (uint8_t i = 0; i < max_tensor_dimensions; i++) {
                indicesSrc[i] = ndx[i];
                if ( i != axis)
                  indicesDest[i] = ndx[i];
                else                  
                  indicesDest[i] = newDims[i]-1-ndx[i];
              }
                
              destH.at(indicesDest, max_tensor_dimensions) =
                srcH.at(indicesSrc, max_tensor_dimensions);
            }
 
}

template <typename ElemTy>
inline void fwdLibFlipInst(LibTensor* inT, LibTensor* outT, unsigned int axis) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  printf("Mid=%d\n",minionId);  

  //@TODO  static_assert(max_tensor_dimensions == 6,"Loops below assume max_tensor_dimensions = 6.");

  dim_t newDims[max_tensor_dimensions] = {0,};
  dim_t currDims[max_tensor_dimensions] = {0,};
  uint8_t numDims =  inT->dims(currDims);
  
  expandDimsToMax(newDims, currDims, numDims);
  
  LibTensor eSrc = inT->getUnowned(newDims, numDims, true);
  LibTensor eDst = outT->getUnowned(newDims, numDims, true);  
  
  auto srcH = eSrc.getHandle<ElemTy>();
  auto destH = eDst.getHandle<ElemTy>();
  
  loopAxis(srcH, destH, newDims, axis);

}

}//namespace inlining

} //namespace dnn_lib

#endif  // _FLIP_INST_H_
