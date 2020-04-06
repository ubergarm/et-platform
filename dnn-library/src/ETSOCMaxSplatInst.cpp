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

#include "ETSOCMaxSplatInst.h" // From include/inlining

namespace dnn_lib {

// This function copies a matrix replacing all the elements which are < splatVal
// and replaces them with splatVal
template <typename srcType>
void fwdLibETSOCMaxSplatInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *srcT,
                                      void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, float splatVal,
                                      const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibETSOCMaxSplatInst<srcType>(dstT, dstDims,
                                      dstPitches, srcT,
                                      srcDims, srcPitches,
                                      srcDimNum, splatVal,
                                      scale, offset);

}

template <typename srcType>
void fwdLibETSOCMaxSplatInst(void *dstT, void *dstDims,
                                      void *dstPitches, void *srcT,
                                      void *srcDims, void *srcPitches,
                                      unsigned int srcDimNum, int64_t splatVal,
                                      const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibETSOCMaxSplatInst<srcType>(dstT, dstDims,
                                      dstPitches, srcT,
                                      srcDims, srcPitches,
                                      srcDimNum, splatVal,
                                      scale, offset);

}

template <typename srcType>
void fwdLibETSOCMaxSplatInstThreaded(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              float splatVal, const float *scale,
                                              const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibETSOCMaxSplatInstThreaded<srcType>(dst, dstDims,
                                              dstPitches, src,
                                              srcDims, srcPitches,
                                              srcDimNum,
                                              splatVal, scale,
                                              offset, flags);
}

template <typename srcType>
void fwdLibETSOCMaxSplatInstThreaded(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              int64_t splatVal, const float *scale,
                                              const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibETSOCMaxSplatInstThreaded<srcType>(dst, dstDims,
                                              dstPitches, src,
                                              srcDims, srcPitches,
                                              srcDimNum,
                                              splatVal, scale,
                                              offset, flags);
}

template <typename srcType>
void fwdLibETSOCMaxSplatInstVectorized(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              float splatVal, const float *scale,
                                              const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibETSOCMaxSplatInstVectorized<srcType>(dst, dstDims,
                                              dstPitches, src,
                                              srcDims, srcPitches,
                                              srcDimNum,
                                              splatVal, scale,
                                              offset, flags);
}

template <typename srcType>
void fwdLibETSOCMaxSplatInstAligned32Bytes(void *dst, void *dstDims,
                                              void *dstPitches, void *src,
                                              void *srcDims, void *srcPitches,
                                              unsigned int srcDimNum,
                                              float splatVal, const float *scale,
                                              const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibETSOCMaxSplatInstAligned32Bytes<srcType>(dst, dstDims,
                                              dstPitches, src,
                                              srcDims, srcPitches,
                                              srcDimNum,
                                              splatVal, scale,
                                              offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInst, void *dstT, void *dstDims, void *dstPitches,
                                void *srcT, void *srcDims, void *srcPitches,
                                unsigned int srcDimNum, float splatVal,
                                const float *scale, const int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInst, void *dstT, void *dstDims, void *dstPitches,
                                void *srcT, void *srcDims, void *srcPitches,
                                unsigned int srcDimNum, int64_t splatVal,
                                const float *scale, const int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInstThreaded,void *dst, void *dstDims, void *dstPitches,
                                         void *src, void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float splatVal,
                                         const float *scale, const int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInstThreaded,void *dst, void *dstDims, void *dstPitches,
                                         void *src, void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, int64_t splatVal,
                                         const float *scale, const int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInstVectorized,void *dst, void *dstDims, void *dstPitches,
                                         void *src, void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float splatVal,
                                         const float *scale, const int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibETSOCMaxSplatInstAligned32Bytes,void *dst, void *dstDims, void *dstPitches,
                                         void *src, void *srcDims, void *srcPitches,
                                         unsigned int srcDimNum, float splatVal,
                                         const float *scale, const int32_t *offset, uint64_t flags);

} // namespace dnn_lib

