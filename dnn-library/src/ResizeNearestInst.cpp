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

#include "ResizeNearestInst.h" // From include/inlining

namespace dnn_lib {


template <ElemKind elKind, size_t N>
void fwdLibResizeNearestInst(LibTensor* outT, LibTensor* inT, std::array<float, N> const &rszScale, uint64_t flags ) {

  dnn_lib::inlining::fwdLibResizeNearestInst<elKind>(outT, inT, rszScale, flags);

}

#include "GenInstances.h"

GEN_INSTANCES_OP_ELK_FLOAT_ARR(fwdLibResizeNearestInst);

} //dnn_lib
