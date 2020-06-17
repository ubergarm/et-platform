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

#include "SpaceToDepthInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elKind>
void fwdLibSpaceToDepthInst(LibTensor* outT, LibTensor* inT, uint32_t blockSize,
			    uint64_t flags) {

  dnn_lib::inlining::fwdLibSpaceToDepthInst<elKind>(outT, inT, blockSize, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP_ELK(template, fwdLibSpaceToDepthInst, LibTensor* outT, 
		     LibTensor* inT, uint32_t blockSize, uint64_t flags);

} // dnn_lib
