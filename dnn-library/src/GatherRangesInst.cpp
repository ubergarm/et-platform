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

#include "GatherRangesInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType, typename indexType>
void fwdLibGatherRangesInst(LibTensor* inT, LibTensor* outT, LibTensor* out2T,
                            LibTensor* rangesT) {

  dnn_lib::inlining::fwdLibGatherRangesInst<srcType, indexType>(inT, outT,
                                                                out2T, rangesT);
}


template <typename srcType, typename indexType>
void fwdLibGatherRangesInstThreaded(LibTensor* inT, LibTensor* outT, LibTensor*out2T,
                                    LibTensor* rangesT, uint64_t flags) {

  dnn_lib::inlining::fwdLibGatherRangesInstThreaded<srcType, indexType>(inT,
                                                                        outT,
                                                                        out2T,
                                                                        rangesT,
                                                                        flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP_INDEX(template, fwdLibGatherRangesInst, LibTensor* inT,
                       LibTensor* outT, LibTensor* out2T, LibTensor* rangesT);

  GEN_INSTANCES_OP_INDEX(template, fwdLibGatherRangesInstThreaded, LibTensor* inT,
                         LibTensor* outT, LibTensor* out2T, LibTensor* rangesT,
                         uint64_t flags);
} // namespace dnn_lib
