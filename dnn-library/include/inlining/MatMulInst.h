/*-------------------------------------------------------------------------
 * Copyright (c) 2025 Ainekko, Co.
 * SPDX-License-Identifier: Apache-2.0
 *-------------------------------------------------------------------------
 */

#ifndef _MATMUL_INST_H_
#define _MATMUL_INST_H_

#include "LibTensor.h"
#include "FullyConnectedInst.h" // From include/inlining path

namespace dnn_lib {

namespace inlining {

// Simply forwards the call to fulyl connected
template <ElemKind elK>
INLINE_ATTR void fwdLibMatMulInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T, const uint64_t flags,
                                  const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  // Forward the call to fully connected without bias
  dnn_lib::inlining::fwdLibFullyConnectedInst<elK, elK> (outT, in1T, in2T, (LibTensor *) nullptr, flags, minionOffset, assignedMinions);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _MATMUL_INST_H_
