/*-------------------------------------------------------------------------
 * Copyright (C) 2020, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#include "CumSumInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elKind>
void fwdLibCumSumInst(LibTensor* outT, LibTensor* inT, bool exclusive, 
		      bool reverse, uint64_t flags) {

  dnn_lib::inlining::fwdLibCumSumInst<elKind>(outT, inT, exclusive, 
					      reverse, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP_ELK(template, fwdLibCumSumInst, LibTensor* outT, 
		     LibTensor* inT, bool exclusive, bool reverse, 
		     uint64_t flags);

} // dnn_lib
