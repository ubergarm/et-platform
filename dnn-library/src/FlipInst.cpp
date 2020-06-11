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

#include "FlipInst.h"  // From include/inlining


namespace dnn_lib {
  
template <ElemKind elK>
void fwdLibFlipInst(LibTensor* outT, LibTensor* inT, unsigned int axis) {

  dnn_lib::inlining::fwdLibFlipInst<elK>(outT, inT, axis);
}

#include "GenInstances.h"
  
GEN_INSTANCES_OP(template, fwdLibFlipInst, LibTensor* outT, LibTensor *inT, unsigned int axis);

} //namespace dnn_lib
