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

#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"
#include "LibTensor.h"

namespace dnn_lib {

namespace inlining {

template <typename ElemTy>
inline void fwdLibFlipInst(LibTensor* inT, LibTensor* outT, unsigned int axis) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

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
