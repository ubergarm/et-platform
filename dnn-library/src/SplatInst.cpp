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

#include "SplatInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elK>
void fwdLibSplatInst(LibTensor* outT, uint64_t *splatValPtr) {
  
  dnn_lib::inlining::fwdLibSplatInst<elK>(outT, splatValPtr);
}

template <ElemKind elK>
void fwdLibSplatInstThreaded(LibTensor* outT, uint64_t *splatValPtr, uint64_t flags) {

  dnn_lib::inlining::fwdLibSplatInstThreaded<elK>(outT, splatValPtr, flags);
}

template <ElemKind elK>
void fwdLibSplatInstVectorized(LibTensor* outT, uint64_t *splatVal, uint64_t flags) {

  dnn_lib::inlining::fwdLibSplatInstVectorized<elK>(outT, splatVal, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSplatInst, LibTensor* outT, uint64_t *splatVal);

GEN_INSTANCES_OP(template, fwdLibSplatInstThreaded, LibTensor* outT, uint64_t *splatVal, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibSplatInstVectorized, LibTensor* outT, uint64_t *splatVal, uint64_t flags);

} // namespace dnn_lib
