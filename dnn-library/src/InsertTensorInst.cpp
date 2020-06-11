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

#include "InsertTensorInst.h"

namespace dnn_lib {

template <ElemKind elK>
void fwdLibInsertTensorInst(LibTensor* outT, LibTensor* inT, const uint32_t *pcoord,
                            unsigned int count, unsigned int axis,
                            uint64_t flags, const uint32_t minionOffset) {

  dnn_lib::inlining::fwdLibInsertTensorInst<elK>(outT, inT, pcoord, count,
                                                     axis, flags, minionOffset);
}

template <ElemKind elK>
void fwdLibInsertTensorInstThreaded(LibTensor* outT, LibTensor* inT,
                                    const uint32_t *poffsets, unsigned int count,
                                    unsigned int axis, uint64_t flags,
                                    const uint32_t minionOffset,
                                    const  uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibInsertTensorInstThreaded<elK>(outT, inT, poffsets,
                                                             count, axis, flags,
                                                             minionOffset,
                                                             assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibInsertTensorInst, LibTensor* outT, LibTensor* inT,
                 const uint32_t *poffsets, unsigned int count, unsigned int axis,
                 uint64_t flags, const uint32_t minionOffset);

GEN_INSTANCES_OP(template, fwdLibInsertTensorInstThreaded, LibTensor* outT,
                 LibTensor* inT, const uint32_t *poffsets, unsigned int count,
                 unsigned int axis, uint64_t flags, const uint32_t minionOffset,
                 const uint32_t assignedMinions);

} // namespace dnn_lib
