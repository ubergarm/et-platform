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

template <typename srcType, typename dstType>
void fwdLibConvertToInst(void *dstT, void *dstDims, void *dstPitches,
                                  void *srcT1, void *srcDims, void *srcPitches,
                                  unsigned int srcDimNum, const float *scale,
                                  const int32_t *offset) {

  dnn_lib::inlining::fwdLibConvertToInst<srcType, dstType>(dstT, dstDims, dstPitches,
                                  srcT1, srcDims, srcPitches,
                                  srcDimNum, scale,
                                  offset);
}

template <typename srcType, typename dstType>
void fwdLibConvertToInstThreaded(void *dst, void *dstDims,
                                          void *dstPitches, void *src,
                                          void *srcDims, void *srcPitches,
                                          unsigned int srcDimNum, const float *scale,
                                          const int32_t *offset, uint64_t flags) {

  dnn_lib::fwdLibConvertToInstThreaded<srcType, dstType>(dst, dstDims,
                                          dstPitches, src,
                                          srcDims, srcPitches,
                                          srcDimNum, scale,
                                          offset, flags);
}

template <typename srcType, typename dstType>
void fwdLibConvertToInstVectorized(void *dst, void *dstDims,
                                            void *dstPitches, void *src,
                                            void *srcDims, void *srcPitches,
                                            unsigned int srcDimNum, const float *scale,
                                            const int32_t *offset, uint64_t flags) {

  dnn_lib::fwdLibConvertToInstVectorized<srcType, dstType>(dst, dstDims,
                                            dstPitches, src,
                                            srcDims, srcPitches,
                                            srcDimNum, scale,
                                            offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_CONVERT(template, fwdLibConvertToInst, void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT1, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, const float *scale, const int32_t *offset);

GEN_INSTANCES_CONVERT(template, fwdLibConvertToInstThreaded, void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT1, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, const float *scale, const int32_t *offset, uint64_t flags);

GEN_INSTANCES_CONVERT(template, fwdLibConvertToInstVectorized, void *dstT, void *dstDims, void *dstPitches,
                                 void *srcT1, void *srcDims, void *srcPitches,
                                 unsigned int srcDimNum, const float *scale, const int32_t *offset, uint64_t flags);
} // namespace dnn_lib
