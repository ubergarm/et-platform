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

#include "TensorViewInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibTensorViewInst(LibTensor* outT, LibTensor* inT, void *pcoord) {

  dnn_lib::inlining::fwdLibTensorViewInst<srcType>(outT, inT, pcoord);
}

template <typename srcType>
void fwdLibTensorViewInstThreaded(LibTensor* outT, LibTensor* inT,
                                  void *pcoord, uint64_t flags) {

  dnn_lib::inlining::fwdLibTensorViewInstThreaded<srcType>(outT, inT,
                                                           pcoord, flags);
}

template <typename srcType>
void fwdLibTensorViewInstVectorized(LibTensor* outT, LibTensor* inT,
                                    void *pcoord, uint64_t flags) {

  dnn_lib::inlining::fwdLibTensorViewInstVectorized<srcType>(outT, inT,
                                                             pcoord, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibTensorViewInst, LibTensor* outT,
                 LibTensor* inT, void *poffsets);

GEN_INSTANCES_OP(template, fwdLibTensorViewInstThreaded, LibTensor* outT,
                LibTensor* inT, void *poffsets, uint64_t flags );

GEN_INSTANCES_OP(template, fwdLibTensorViewInstVectorized, LibTensor* outT,
                 LibTensor* inT, void *poffsets, uint64_t flags );
  
} // namespace dnn_lib
