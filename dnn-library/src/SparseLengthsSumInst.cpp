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

#include "SparseLengthsSumInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elKind>
void fwdLibSparseLengthsSumInst(LibTensor* outT, LibTensor* inT, LibTensor* in2T,
				LibTensor* in3T, uint64_t flags) {

  dnn_lib::inlining::fwdLibSparseLengthsSumInst<elKind>(outT, inT, in2T, in3T, flags);
}


template <ElemKind elKind>
void fwdLibSparseLengthsSumInstThreaded(LibTensor* outT, LibTensor* in1T, 
					LibTensor* in2T, LibTensor* in3T,                                        
					uint64_t flags) {

  dnn_lib::inlining::fwdLibSparseLengthsSumInstThreaded<elKind>(outT, in1T, in2T, 
								in3T, flags);
}


#include "GenInstances.h"

GEN_INSTANCES_OP_ELK(template, fwdLibSparseLengthsSumInst, LibTensor* outT,
		     LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, 
		     uint64_t flags);

GEN_INSTANCES_OP_ELK(template, fwdLibSparseLengthsSumInstThreaded, 
		     LibTensor* outT, LibTensor* in1T, LibTensor* in2T, 
		     LibTensor* in3T, uint64_t flags);


}
