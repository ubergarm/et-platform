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

#include "ElementImmLogic.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elK, typename opType>
void fwdLibElementImmLogic(LibTensor* outT, LibTensor* inT, void *imm) {

  dnn_lib::inlining::fwdLibElementImmLogic<elK, opType>(outT, inT, imm);
}

#include "GenInstances.h"

GEN_INSTANCES_INSTANCES_LOGIC(template, fwdLibElementImmLogic, And, LibTensor* outT,
                              LibTensor* inT, void *imm);

GEN_INSTANCES_INSTANCES_LOGIC(template, fwdLibElementImmLogic, Or, LibTensor* outT,
                              LibTensor* inT, void *imm);

GEN_INSTANCES_INSTANCES_LOGIC(template, fwdLibElementImmLogic, Xor, LibTensor* outT,
                              LibTensor* inT, void *imm);

} // namespace dnn_lib
