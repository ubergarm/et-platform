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

#ifndef _MATMUL_INST_H_
#define _MATMUL_INST_H_

#include "LibTensor.h"
#include "MatMulInstTransposed.h" // From include/inlining path
#include "FullyConnectedInst.h" // From include/inlining path

namespace dnn_lib {

namespace inlining {

// Simply forwards the call to fulyl connected
template <ElemKind elK>
inline void fwdLibMatMulInst(LibTensor* outT, LibTensor* in1T,
                                LibTensor* in2T, bool transposed,
                                const uint64_t flags,
                                const uint32_t minionOffset = 0,
                                const uint32_t assignedMinions = 0) {
  if (transposed) return  fwdLibMatMulInstTransposed<elK>(outT, in1T, in2T, flags, minionOffset, assignedMinions);

  // Forward the call to fully connected without bias
  dnn_lib::inlining::fwdLibFullyConnectedInst<elK, elK> (outT, in1T, in2T, (LibTensor *) nullptr, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _MATMUL_INST_H_
