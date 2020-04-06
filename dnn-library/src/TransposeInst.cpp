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
void fwdLibTransposeInst(void *dst, void *dstDims, void *dstPitches,
                                  void *src, void *srcDims, void *srcPitches,
                                  unsigned int srcDimNum, void *pshuffle,
                                  const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibTransposeInst<srcType>(dst, dstDims, dstPitches,
                                  src, srcDims, srcPitches,
                                  srcDimNum, pshuffle,
                                  scale, offset);
}

template <typename srcType>
void fwdLibTransposeInstThreaded(void *dst, void *dstDims,
                                          void *dstPitches, void *src,
                                          void *srcDims, void *srcPitches,
                                          unsigned int srcDimNum,
                                          void *pshuffle, const float *scale,
                                          const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibTransposeInstThreaded<srcType>(dst, dstDims,
                                          dstPitches, src,
                                          srcDims, srcPitches,
                                          srcDimNum,
                                          pshuffle, scale,
                                          offset, flags);
}

template <typename srcType>
void fwdLibTransposeInstVectorized(void *dst, void *dstDims,
                                            void *dstPitches, void *src,
                                            void *srcDims, void *srcPitches,
                                            unsigned int srcDimNum,
                                            void *pshuffle, const float *scale,
                                            const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibTransposeInstVectorized<srcType>(dst, dstDims,
                                            dstPitches, src,
                                            srcDims, srcPitches,
                                            srcDimNum,
                                            pshuffle, scale,
                                            offset, flags);
}

template <typename srcType>
void fwdLibTransposeInstAligned32Bytes(void *dst,
                                          void *dstDims,
                                          void *dstPitches, void *src,
                                          void *srcDims, void *srcPitches,
                                          unsigned int srcDimNum,
                                          void *pshuffle, const float *scale,
                                          const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibTransposeInstAligned32Bytes<srcType>(dst,
                                          dstDims,
                                          dstPitches, src,
                                          srcDims, srcPitches,
                                          srcDimNum,
                                          pshuffle, scale,
                                          offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibTransposeInst, void *dst, void *dstDims, void *dstPitches,
                            void *src, void *srcDims, void *srcPitches,
                            unsigned int srcDimNum, void *pshuffle,
                            const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibTransposeInstThreaded, void *dst, void *dstDims, void *dstPitches,
                            void *src, void *srcDims, void *srcPitches,
                            unsigned int srcDimNum, void *pshuffle,
                            const float *scale, const int32_t *offset, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibTransposeInstVectorized, void *dst, void *dstDims, void *dstPitches,
                            void *src, void *srcDims, void *srcPitches,
                            unsigned int srcDimNum, void *pshuffle,
                            const float *scale, const int32_t *offset, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibTransposeInstAligned32Bytes, void *dst, void *dstDims, void *dstPitches,
                            void *src, void *srcDims, void *srcPitches,
                            unsigned int srcDimNum, void *pshuffle,
                            const float *scale, const int32_t *offset, uint64_t flags);
} // namespace dnn_lib
