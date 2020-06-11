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

#include "TopKInst.h" // From include/inlining

namespace dnn_lib {

void partialQuicksort(void *vals, void *inds, int low, int high, int m) {
  if (low < high) {
    int pidx = dnn_lib::inlining::partition(vals, inds, low, high);
    partialQuicksort(vals, inds, low, pidx - 1, m);
    if (pidx < m) {
      partialQuicksort(vals, inds, pidx + 1, high, m);
    }
  }
}

template <ElemKind elK>
void fwdLibTopKInst(LibTensor* outT, LibTensor* out2T, LibTensor* inT,
                    unsigned int k) {

  dnn_lib::inlining::fwdLibTopKInst<elK>(outT, out2T, inT, k);
}

template <ElemKind elK>
void fwdLibTopKInstThreaded_all(LibTensor* outT, LibTensor* out2T,
                                LibTensor* inT, unsigned int k, uint64_t flags) {

  dnn_lib::inlining::fwdLibTopKInstThreaded_all<elK>(outT, out2T, inT, k, flags);
}

template <ElemKind elK>
void fwdLibTopKInstThreaded_k4(LibTensor* outT, LibTensor* out2T,
                               LibTensor* inT, unsigned int k, uint64_t flags) {

  dnn_lib::fwdLibTopKInstThreaded_k4<elK>(outT, out2T, inT, k, flags);
}

template <ElemKind elK>
void fwdLibTopKInstThreaded_k8(LibTensor* outT, LibTensor* out2T,
                               LibTensor* inT, unsigned int k, uint64_t flags) {

  dnn_lib::inlining::fwdLibTopKInstThreaded_k8<elK>(outT, out2T, inT, k, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibTopKInst, LibTensor* outT, LibTensor* out2T,
                 LibTensor* inT, unsigned int k);

GEN_INSTANCES_OP(template, fwdLibTopKInstThreaded_all, LibTensor* outT, LibTensor* out2T,
                 LibTensor* inT, unsigned int k, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibTopKInstThreaded_k4, LibTensor* outT, LibTensor* out2T,
                 LibTensor* inT, unsigned int k, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibTopKInstThreaded_k8, LibTensor* outT, LibTensor* out2T,
                 LibTensor* inT, unsigned int k, uint64_t flags);
} // namespace dnn_lib
