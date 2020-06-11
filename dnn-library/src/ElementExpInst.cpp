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

#include "ElementExpInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind elK>
void fwdLibElementExpInst(LibTensor* outT, LibTensor* inT) {

  dnn_lib::inlining::fwdLibElementExpInst<elK>(outT, inT);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibElementExpInst, LibTensor* outT, LibTensor* inT);
} // namespace dnn_lib
