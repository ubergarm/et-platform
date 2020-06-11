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

#include "ConvertToInst.h" // From include/inlining

namespace dnn_lib {

template <ElemKind dstElK, ElemKind srcElK>
void fwdLibConvertToInst(LibTensor* inT, LibTensor* outT) {

  dnn_lib::inlining::fwdLibConvertToInst<dstElK, srcElK>(inT, outT);
}

template <ElemKind dstElK, ElemKind srcElK>
void fwdLibConvertToInstThreaded(LibTensor* inT, LibTensor* outT, uint64_t flags) {

  dnn_lib::fwdLibConvertToInstThreaded<dstElK, srcElK>(inT, outT, flags);
}

template <ElemKind dstElK, ElemKind srcElK>
void fwdLibConvertToInstVectorized(LibTensor* inT, LibTensor* outT, uint64_t flags) {

  dnn_lib::fwdLibConvertToInstVectorized<dstElK, srcElK>(inT, outT, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_CONVERT(template, fwdLibConvertToInst, LibTensor* inT, LibTensor* outT);

GEN_INSTANCES_CONVERT(template, fwdLibConvertToInstThreaded, LibTensor* inT, LibTensor* outT, uint64_t flags);

GEN_INSTANCES_CONVERT(template, fwdLibConvertToInstVectorized, LibTensor* inT, LibTensor* outT, uint64_t flags);
} // namespace dnn_lib
