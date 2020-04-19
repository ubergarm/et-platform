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

#include "TransposeInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibTransposeInst(LibTensor* outT, LibTensor* inT, void* pshuffle) {

  dnn_lib::inlining::fwdLibTransposeInst<srcType>(outT, inT, pshuffle);
}

template <typename srcType>
void fwdLibTransposeInstThreaded(LibTensor* outT, LibTensor* inT,
                                 void* pshuffle, uint64_t flags) {

  dnn_lib::inlining::fwdLibTransposeInstThreaded<srcType>(outT, inT, pshuffle,
                                                          flags);
}

template <typename srcType>
void fwdLibTransposeInstVectorized(LibTensor* outT, LibTensor* inT,
                                   void* pshuffle, uint64_t flags) {

  dnn_lib::inlining::fwdLibTransposeInstVectorized<srcType>(outT, inT,
                                                            pshuffle, flags);
}

template <typename srcType>
void fwdLibTransposeInstAligned32Bytes(LibTensor* outT, LibTensor* inT,
                                          void* pshuffle, uint64_t flags) {

  dnn_lib::inlining::fwdLibTransposeInstAligned32Bytes<srcType>(outT, inT,
                                                                pshuffle, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibTransposeInst, LibTensor* outT,
                 LibTensor* inT, void* pshuffle);

GEN_INSTANCES_OP(template, fwdLibTransposeInstThreaded, LibTensor* outT,
                 LibTensor* inT, void* pshuffle, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibTransposeInstVectorized, LibTensor* outT,
                   LibTensor* inT, void* pshuffle, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibTransposeInstAligned32Bytes, LibTensor* outT,
                 LibTensor* inT, void* pshuffle, uint64_t flags);
  
} // namespace dnn_lib
