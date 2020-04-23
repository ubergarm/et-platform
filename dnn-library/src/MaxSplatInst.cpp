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

#include "MaxSplatInst.h" // From include/inlining

namespace dnn_lib {

// This function copies a matrix replacing all the elements which are < splatVal
// and replaces them with splatVal
template <typename srcType>
void fwdLibMaxSplatInst(LibTensor* outT, LibTensor* inT, float splatVal) {

  dnn_lib::inlining::fwdLibMaxSplatInst<srcType>(outT, inT, splatVal);

}

template <typename srcType>
void fwdLibMaxSplatInst(LibTensor* outT, LibTensor* inT, int64_t splatVal) {

  dnn_lib::inlining::fwdLibMaxSplatInst<srcType>(outT, inT, splatVal);

}

template <typename srcType>
void fwdLibMaxSplatInstThreaded(LibTensor* outT, LibTensor* inT, float splatVal, uint64_t flags) {

  dnn_lib::inlining::fwdLibMaxSplatInstThreaded<srcType>(outT, inT, splatVal, flags);
}

template <typename srcType>
void fwdLibMaxSplatInstThreaded(LibTensor* outT, LibTensor* inT, int64_t splatVal, uint64_t flags) {

  dnn_lib::inlining::fwdLibMaxSplatInstThreaded<srcType>(outT, inT, splatVal, flags);
}

template <typename srcType>
void fwdLibMaxSplatInstVectorized(LibTensor* outT, LibTensor* inT, float splatVal, uint64_t flags) {

  dnn_lib::inlining::fwdLibMaxSplatInstVectorized<srcType>(outT, inT, splatVal, flags);
}

template <typename srcType>
void fwdLibMaxSplatInstAligned32Bytes(LibTensor* outT, LibTensor* inT, float splatVal, uint64_t flags) {

  dnn_lib::inlining::fwdLibMaxSplatInstAligned32Bytes<srcType>(outT, inT, splatVal, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibMaxSplatInst, LibTensor* outT, LibTensor* inT, float splatVal);
GEN_INSTANCES_OP(template, fwdLibMaxSplatInst, LibTensor* outT, LibTensor* inT, int64_t splatVal);
GEN_INSTANCES_OP(template, fwdLibMaxSplatInstThreaded, LibTensor* outT, LibTensor* inT, float splatVal, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibMaxSplatInstThreaded, LibTensor* outT, LibTensor* inT, int64_t splatVal, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibMaxSplatInstVectorized, LibTensor* outT, LibTensor* inT, float splatVal, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibMaxSplatInstAligned32Bytes, LibTensor* outT, LibTensor* inT, float splatVal, uint64_t flags);

} // namespace dnn_lib

