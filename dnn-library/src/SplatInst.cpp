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

#include "SplatInst.h" // From include/inlining

namespace dnn_lib {

template <typename srcType>
void fwdLibSplatInst(void *dst, void *dstDims,
                     void *dstPitches, unsigned int dstDimNum,
                     uint64_t *splatValPtr, const float *scale,
                     const int32_t *offset, uint64_t flags) {
  
  dnn_lib::inlining::fwdLibSplatInst<srcType>
    (dst,dstDims, dstPitches, dstDimNum,
     splatValPtr, scale, offset, flags
     );
}

template <typename sourceTy>
void fwdLibSplatInstThreaded(void *dst, void *dstDims,
                                      void *dstPitches, unsigned int dstDimNum,
                                      uint64_t *splatValPtr, const float *scale,
                                      const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibSplatInstThreaded<sourceTy>(dst, dstDims,
                                      dstPitches, dstDimNum,
                                      splatValPtr, scale,
                                      offset, flags);
}

template <typename srcType>
void fwdLibSplatInstVectorized(void *dst, void *dstDims,
                                        void *dstPitches, unsigned int dstDimNum,
                                        uint64_t *splatVal, const float *scale,
                                        const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibSplatInstVectorized<srcType>(dst, dstDims,
                                        dstPitches, dstDimNum,
                                        splatVal, scale,
                                        offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibSplatInst,  void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, uint64_t *splatVal,
                             const float *scale, const int32_t *offset, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibSplatInstThreaded, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, uint64_t *splatVal,
                             const float *scale, const int32_t *offset, uint64_t flags);

GEN_INSTANCES_OP(template, fwdLibSplatInstVectorized, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, uint64_t *splatVal,
                             const float *scale, const int32_t *offset, uint64_t flags);

} // namespace dnn_lib
