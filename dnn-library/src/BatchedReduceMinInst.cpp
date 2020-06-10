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

#include "BatchedReduceMinInst.h" // From include/inlining

namespace dnn_lib {


template <ElemKind elKind, size_t N>
void fwdLibBatchedReduceMinInst(LibTensor* outT, LibTensor* inT, std::array<uint32_t, N> const &axes, uint64_t flags ) {

  dnn_lib::inlining::fwdLibBatchedReduceMinInst<elKind>(outT, inT, axes, flags);

}

#include "GenInstances.h"

GEN_INSTANCES_OP_ELK_UINT32_ARR(fwdLibBatchedReduceMinInst);

} //dnn_lib


