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

#ifndef _SPARSE_LENGTHS_SUM_INST_H
#define _SPARSE_LENGTHS_SUM_INST_H

#include "LibTypes.h"

#include "SparseLengthsWeightedSumInst.h"

namespace dnn_lib {
namespace inlining {

template <ElemKind elKind>
INLINE_ATTR void fwdLibSparseLengthsSumInst(LibTensor* outT, LibTensor* in1T,
					    LibTensor* in2T, LibTensor* in3T, 					    
					    uint64_t flags) {

  LibTensor* inW = nullptr;

  dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInst<elKind>(outT, in1T, 
                                                       inW, in2T, in3T, 
			   	   		       flags);

}
 
template <ElemKind elKind>
INLINE_ATTR void fwdLibSparseLengthsSumInstThreaded(LibTensor* outT, 
						    LibTensor* in1T,
						    LibTensor* in2T,
						    LibTensor* in3T, 
						    uint64_t flags) {

  LibTensor* inW = nullptr;

  dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInstThreaded<elKind>(outT, 
                                                        in1T, inW, in2T, in3T,
					                flags);
}

} // inlining
} // namespace

#endif
