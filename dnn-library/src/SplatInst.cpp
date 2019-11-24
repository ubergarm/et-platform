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

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "LibNodes.h"
#include "GenInstances.h"
#include "Float16.h"
#include "Writer.h"
#include "Addresser.h"
#include "Converter.h"
#include "Operator.h"
#include "utils.h"

using namespace std;

template <typename srcType>
void dnn_lib::fwdLibSplatInst(void *addr, int numElems, float splatVal,
                              float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(addr, scale[0], offset[0]);
  for (int i = 0; i < numElems; i++) {
    tOutput[i] = splatVal;
  }
}

template <typename srcType>
void dnn_lib::fwdLibSplatInst(void *addr, int numElems, int64_t splatVal,
                              float *scale, int32_t *offset) {
  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  srcType *tOutput = (srcType *)addr;
  for (int i = 0; i < numElems; i++) {
    tOutput[i] = splatVal;
  }
}

template <typename srcType>
void dnn_lib::fwdLibSplatInstThreaded(void *dst, void *dstDims,
                                      void *dstPitches, unsigned int dstDimNum,
                                      float splatVal, float *scale,
                                      int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

 Addresser<srcType> tOutput(dst, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  // Get minion id
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[dstDimNum]; // Vector of coordinates
  unsigned int k;                  // Amount of non-zero coordinates

  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    tOutput[offsetOut] = splatVal;
    done = getOffsets(dstDimNum, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibSplatInstThreaded(void *dst, void *dstDims,
                                      void *dstPitches, unsigned int dstDimNum,
                                      int64_t splatVal, float *scale,
                                      int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  srcType *tOutput = (srcType *)dst;

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];

  // Get minion id
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;
  unsigned int coord[dstDimNum]; // Vector of coordinates
  unsigned int k;                  // Amount of non-zero coordinates

  getNonPaddingCoordinates(coord, initialAddr, dstDimNum, dstPitch, dstIndex,
                           k);

  // We get the actual initialAddr, in the input and output.
  unsigned int offsetOut = 0;
  for (unsigned int j = 0; j < k; j++) {
    offsetOut += dstPitch[j] * coord[j];
  }
  unsigned int posMax = maxRead + initialAddr;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    tOutput[offsetOut] = splatVal;
    done = getOffsets(dstDimNum, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

template <typename srcType>
void dnn_lib::fwdLibSplatInstVectorized(void *dst, void *dstDims,
                                        void *dstPitches, unsigned int dstDimNum,
                                        float splatVal, float *scale,
                                        int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dst, scale[1], offset[1]);

  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  size_t typeSize = getsize<srcType>();
  size_t bytesperCL = 64;

  uint64_t totalBytes = dstPitch[0] * dstIndex[0] * typeSize;
  uint64_t totalCL = (totalBytes - 1)/bytesperCL + 1;
  uint64_t CLperMinion = (totalCL - 1)/activeMinions + 1;
  uint64_t startCL = minionId * CLperMinion;

  if (startCL >= totalCL) return;
  if (startCL + CLperMinion > totalCL) CLperMinion = totalCL - startCL;

  uint64_t regsperMinion = 2 * CLperMinion;  // A cacheline contains 2 regs
  uint64_t offsetOut = startCL * 64;
  uint64_t startElem = offsetOut/typeSize;
  char *dstPtr = (char *)dst;
  dstPtr += offsetOut;

  size_t numVals = 8/typeSize;
  for (unsigned int j = 0; j < numVals; j++)
    tOutput[startElem + j] = splatVal;

  __asm__ __volatile__("mov.m.x m0, zero, 0x55\n"
                       "fbc.ps f0, 0x0(%[dstPtr])\n"
                       "mov.m.x m0, zero, 0xaa\n"
                       "fbc.ps f0, 0x4(%[dstPtr])\n"
                       "mov.m.x m0, zero, 0xff\n"

                       "beq %[regs], zero, 1f\n"
                       "2:\n"
                       "fsw.ps f0, 0x0(%[dstPtr])\n"
                       "addi %[dstPtr], %[dstPtr], 0x20\n"
                       "addi %[regs], %[regs], -1\n"
                       "bne %[regs], zero, 2b\n"
                       "1:\n"

                       : [dstPtr] "+r"(dstPtr),
                         [regs] "+r"(regsperMinion)
                       :
                       : "f0");
}

template <typename srcType>
void dnn_lib::fwdLibSplatInstVectorized(void *dst, void *dstDims,
                                        void *dstPitches, unsigned int dstDimNum,
                                        int64_t splatVal, float *scale,
                                        int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();

  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  srcType *tOutput = (srcType *)dst;
  unsigned int *dstIndex = (unsigned int *)dstDims;
  unsigned int *dstPitch = (unsigned int *)dstPitches;
  size_t typeSize = getsize<srcType>();
  size_t bytesperCL = 64;

  uint64_t totalBytes = dstPitch[0] * dstIndex[0] * typeSize;
  uint64_t totalCL = (totalBytes - 1)/bytesperCL + 1;
  uint64_t CLperMinion = (totalCL - 1)/activeMinions + 1;
  uint64_t startCL = minionId * CLperMinion;

  if (startCL >= totalCL) return;
  if (startCL + CLperMinion > totalCL) CLperMinion = totalCL - startCL;

  uint64_t regsperMinion = 2 * CLperMinion;  // A cacheline contains 2 regs
  uint64_t offsetOut = startCL * 64;
  uint64_t startElem = offsetOut/typeSize;
  char *dstPtr = (char *)dst;
  dstPtr += offsetOut;

  size_t numVals = 8/typeSize;
  for (unsigned int j = 0; j < numVals; j++)
    tOutput[startElem + j] = splatVal;

  __asm__ __volatile__("mov.m.x m0, zero, 0x55\n"
                       "fbc.ps f0, 0x0(%[dstPtr])\n"
                       "mov.m.x m0, zero, 0xaa\n"
                       "fbc.ps f0, 0x4(%[dstPtr])\n"
                       "mov.m.x m0, zero, 0xff\n"

                       "beq %[regs], zero, 1f\n"
                       "2:\n"
                       "fsw.ps f0, 0x0(%[dstPtr])\n"
                       "addi %[dstPtr], %[dstPtr], 0x20\n"
                                   "addi %[regs], %[regs], -1\n"
                       "bne %[regs], zero, 2b\n"
                       "1:\n"

                       : [dstPtr] "+r"(dstPtr),
                         [regs] "+r"(regsperMinion)
                       :
                       : "f0");
}

GEN_INSTANCES_OP(template, fwdLibSplatInst, void *addr, int numElems, float splatVal,
                        float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibSplatInst, void *addr, int numElems, int64_t splatVal,
                        float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibSplatInstThreaded, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, float splatVal,
                             float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibSplatInstThreaded, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, int64_t splatVal,
                             float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibSplatInstVectorized, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, float splatVal,
                             float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibSplatInstVectorized, void *dst, void *dstDims, void *dstPitches,
                             unsigned int dstDimNum, int64_t splatVal,
                             float *scale, int32_t *offset, uint64_t flags);
