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

#ifndef _ARG_MAX_INST_H_
#define _ARG_MAX_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "Operators.h"
#include "Float16.h"
#include "Writer.h" // From include/internal path
#include "Addresser.h" // From include/internal path
#include "Converter.h" // From include/internal path
#include "Operator.h" // From include/internal path
#include "utils.h" // From include/internal path

using namespace std;

namespace dnn_lib {

namespace inlining {

template <typename srcType>
inline void fwdLibArgMaxInst( void *src, void *srcDims, void *srcPitches, float srcScale, int32_t srcOffset,
                                void *dst, void *dstDims, void *dstPitches,
                                size_t axis, bool keepDim){
  // cast src parameters to objects we can handle
  const Addresser<srcType> in(src, srcScale, srcOffset);
  unsigned int *inPitch = static_cast<unsigned int *>(srcPitches);
  unsigned int *inDims = static_cast<unsigned int *>(srcDims);

  // cast dst parameters to objects we can handle
  sdim_t *argmax = static_cast<sdim_t*> (dst);
  unsigned int *argmaxPitch = static_cast<unsigned int *>(dstPitches);
  unsigned int *argmaxDims = static_cast<unsigned int *>(dstDims);

  // and compute
  dim_t a, b, c, d = 0;

  dim_t *dim[4];
  dim[(axis + 1) % 4] = &a;
  dim[(axis + 2) % 4] = &b;
  dim[(axis + 3) % 4] = &c;
  dim[axis] = &d;

  // if keepDim == false, pitches and dimensions are 4 => select output pitch setting 0 in the reduced dimension
  unsigned outPitch[4];
  if (!keepDim) {
    for(unsigned i = 0 ; i < axis; i++) outPitch[i] = argmaxPitch[i];
    outPitch[axis] = 0;
    for(unsigned i = axis+1 ; i < 4; i++) outPitch[i] = argmaxPitch[i-1];
    argmaxPitch = outPitch;
  }

  
  for (a = 0; a < inDims[(axis + 1) % 4]; a++) {
    for (b = 0; b < inDims[(axis + 2) % 4]; b++) {
      for (c = 0; c < inDims[(axis + 3) % 4]; c++) {
        
        float max = in[*(dim[0]) * inPitch[0] + 
                       *(dim[1]) * inPitch[1] +
                       *(dim[2]) * inPitch[2]];
        dim_t maxi = 0;
        
        for (d = 0; d < inDims[axis]; d++) {
          float elem = in[*(dim[0]) * inPitch[0] +
                          *(dim[1]) * inPitch[1] +
                          *(dim[2]) * inPitch[2] +
                          *(dim[3]) * inPitch[3]];
          if (elem > max) {
            max = elem;
            maxi = d;
          }
        }
        *dim[axis] = 0;
        dim_t ind = (*dim[0]) * argmaxPitch[0] + 
          (*dim[1]) * argmaxPitch[1] +
          (*dim[2]) * argmaxPitch[2] +
          (*dim[3]) *  argmaxPitch[3] ;
        argmax[ind] = maxi;
      }
    }
  }
  
}

} // namespace dnn_lib

} // namespace inlininig

#endif // _ARG_MAX_INST_H_
