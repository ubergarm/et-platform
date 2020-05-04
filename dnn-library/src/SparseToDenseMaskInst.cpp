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

#include "SparseToDenseMaskInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibSparseToDenseMaskInst(LibTensor* outT, LibTensor* in1T,
                                 LibTensor* in2T, LibTensor* in3T,
                                 LibTensor* in4T, unsigned int pdefaultSize,
                                 unsigned int lengthsSize,
                                 void *pmask, unsigned int pMaskSize) {

  dnn_lib::inlining::fwdLibSparseToDenseMaskInst<srcType>(outT, in1T, in2T,
                                                          in3T, in4T,
                                                          pdefaultSize,
                                                          lengthsSize,
                                                          pmask, pMaskSize);
}

template <typename srcType>
void fwdLibSparseToDenseMaskInstThreaded(LibTensor* outT, LibTensor* in1T,
                                         LibTensor* in2T, LibTensor* in3T,
                                         LibTensor* in4T, unsigned int pdefaultSize,
                                         unsigned int lengthsSize,
                                         void *pmask, unsigned int pMaskSize,
                                         uint64_t flags) {

  dnn_lib::inlining::fwdLibSparseToDenseMaskInstThreaded<srcType>(outT, in1T, in2T,
                                                                  in3T, in4T,
                                                                  pdefaultSize,
                                                                  lengthsSize,
                                                                  pmask, pMaskSize,
                                                                  flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSparseToDenseMaskInst, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                 LibTensor* in4T, unsigned int pdefaultSize,
                 unsigned int lengthsSize, void *pmask, unsigned int pMaskSize);

GEN_INSTANCES_OP(template, fwdLibSparseToDenseMaskInstThreaded, LibTensor* outT,
                 LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                 LibTensor* in4T,  unsigned int pdefaultSize, unsigned int lengthsSize, void *pmask,
                 unsigned int pMaskSize, uint64_t flags);

} // namespace dnn_lib
