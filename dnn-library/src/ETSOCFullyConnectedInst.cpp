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

template <typename srcType>
void dnn_lib::fwdLibETSOCFullyConnectedInst(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    float *scale, int32_t *offset) {

  unsigned int minionId = get_minion_id();
  if (minionId != 0)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  float *tBias = (float *)bias;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Add> opAdd;
  Operator<Addresser<srcType>, Addresser<srcType>, Addresser<srcType>, Mul> opMul;
  uint64_t addrSrc1, addrSrc2, addrSrc3, addrDst;
  // For each (x,y) in the destination matrix:
  for (unsigned int x = 0; x < dstIndex[0]; x++) {
    for (unsigned int y = 0; y < dstIndex[1]; y++) {
      // Perform DOT on the row an column.
      float sum = 0.0;
      for (unsigned int i = 0; i < actIndex[1]; i++) {
        sum += float(tAInput[x * actPitch[0] + i] *
                     tWInput[i * weightPitch[0] + y]);
      }
      sum += tBias[y];
      tOutput[x * dstPitch[0] + y] = sum;
    }
  }
}

template <typename srcType>
void dnn_lib::fwdLibETSOCFullyConnectedInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  float *tBias = (float *)bias;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[2] = {0, 0};
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 2, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  while (!done && (offsetOut < posMax)) {
    float sum = 0;
    for (unsigned int i = 0; i < actIndex[1]; i++) {
      sum += float(tAInput[coord[0] * actPitch[0] + i]) *
             float(tWInput[i * weightPitch[0] + coord[1]]);
    }
    sum += tBias[coord[1]];
    tOutput[offsetOut] = float(sum);
    done = getOffsets(2, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, float>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

#define MATMUL_ITERATION               \
    "flw.ps   f0, 0x0(%[actAddr])\n"   \
    "fgw.ps   f1, f29(%[wgtAddr])\n"   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x20\n"
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"

    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f30, 0x0(%[biasAddr])\n"
    "fadd.s f31, f30, f31\n"
    "fsw.ps f31, 0x0(%[dstAddr])\n"

    :
    : [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr)
    : "t0", "t1", "f0", "f1", "f29", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, float16>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

#define MATMUL_ITERATION               \
    "fgh.ps   f0, f28(%[actAddr])\n"   \
    "fgh.ps   f1, f29(%[wgtAddr])\n"   \
    "fcvt.ps.f16 f0, f0 \n"            \
    "fcvt.ps.f16 f1, f1 \n"            \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f28, 0x0(%[gthValuesAct])\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x10\n"
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"

    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f30, 0x0(%[biasAddr])\n"
    "fadd.s f31, f30, f31\n"
    "fcvt.f16.ps f31, f31\n"           // Conversion fp32 >> fp16.
    "fsch.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr)
    : "t0", "t1", "f0", "f1", "f28", "f29", "f30", "f31", "memory");

#undef MATMUL_ITERATION
}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

#define INT8_TO_FP32(_reg)                  \
    "fsub.pi " #_reg ", " #_reg ", f26 \n"  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"    \
    "fmul.ps " #_reg ", " #_reg ", f27 \n"

#define MATMUL_ITERATION               \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f26, 0x0(%[offset]) \n"  \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                   \
    "fbc.ps   f26, 0x4(%[offset]) \n"  \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define FP32_TO_INT8(_reg)                      \
    "frcp.ps f27, f27 \n"                       \
    "fcvt.ps.pw f26, f26 \n"                    \
    "fmadd.ps " #_reg ", " #_reg ", f27, f26 \n" \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsat8.pi " #_reg ", " #_reg "\n"

  __asm__ __volatile__(
    "mov.m.x m0, zero, 0xff\n"
    "xor t0, t0, t0\n"
    "flw.ps f28, 0x0(%[gthValuesAct])\n"
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"
    "fxor.pi f31, f31, f31\n"

    "1:\n"
    "addi     t0, t0, 8\n"
    "ble      %[elemsRow], t0, 2f\n"
    MATMUL_ITERATION
    "addi %[actAddr], %[actAddr], 0x8\n"
    "fadd.pi f29, f29, f30\n"
    "beq      zero, zero, 1b\n"

    "2:\n"
    "fxor.pi  f0, f0, f0\n"
    "addi     t0, t0, -8\n"
    "sub      t0, %[elemsRow], t0\n"
    "addi     t1, zero, 1\n"
    "sll      t1, t1, t0\n"
    "addi     t1, t1, -1\n"
    "mov.m.x  m0, t1, 0\n"
    MATMUL_ITERATION
    "fmvs.x.ps t0, f31, 0x4\n"
    "fmv.w.x   f30, t0\n"
    "fadd.s    f31, f30, f31\n"
    "mov.m.x m0, zero, 0x1\n"
    "fbc.ps f30, 0x0(%[biasAddr])\n"
    "fadd.s f31, f30, f31\n"
    "fbc.ps f26, 0x8(%[offset]) \n"
    "fbc.ps f27, 0x8(%[scale]) \n"
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

#undef INT8_TO_FP32
#undef MATMUL_ITERATION
#undef FP32_TO_INT8
}

#define INT8_TO_FP32(_reg)                  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"    \
    "fmul.ps " #_reg ", " #_reg ", f27 \n"

#define UINT8_TO_FP32(_reg)                   \
    "fandi.pi " #_reg ", " #_reg ", 0xff \n"  \
    "fcvt.ps.pw " #_reg ", " #_reg " \n"      \
    "fmul.ps " #_reg ", " #_reg ", f27 \n"


#define MATMUL_ITERATION_U8_U8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    UINT8_TO_FP32(f0)                   \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    UINT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define MATMUL_ITERATION_I8_U8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                   \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    UINT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define MATMUL_ITERATION_U8_I8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    UINT8_TO_FP32(f0)                   \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"

#define MATMUL_ITERATION_I8_I8         \
    "fgb.ps   f0, f28(%[actAddr])\n"   \
    "fgb.ps   f1, f29(%[wgtAddr])\n"   \
    "fbc.ps   f27, 0x0(%[scale]) \n"   \
    INT8_TO_FP32(f0)                   \
    "fbc.ps   f27, 0x4(%[scale]) \n"   \
    INT8_TO_FP32(f1)                   \
    "fmul.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0xe\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fswizz.ps  f1, f0, 0x1\n"         \
    "fadd.ps    f0, f0, f1\n"          \
    "fadd.ps    f31, f0, f31\n"


#define FP32_TO_INT8(_reg)                      \
    "frcp.ps f27, f27 \n"                       \
    "fmadd.ps " #_reg ", " #_reg ", f27, f26 \n" \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsat8.pi " #_reg ", " #_reg "\n"


#define FP32_TO_UINT8(_reg)                     \
    "frcp.ps f27, f27 \n"                       \
    "fmul.ps " #_reg ", " #_reg ", f27 \n"      \
    "fcvt.pw.ps " #_reg ", " #_reg "\n"         \
    "fsrli.pi f2," #_reg ", 0x8 \n"             \
    "fxor.pi f27, f27, f27 \n"                  \
    "fcmov.ps " #_reg" , f12, f27, " #_reg " \n"


#define STEP1                                            \
    "mov.m.x m0, zero, 0xff\n"                           \
    "xor t0, t0, t0\n"                                   \
    "flw.ps f28, 0x0(%[gthValuesAct])\n"                 \
    "flw.ps f29, 0x0(%[gthValuesWgt])\n"                 \
    "fbc.ps f30, 0x0(%[wgtRegStep])\n"                   \
    "fxor.pi f31, f31, f31\n"                            \
                                                         \
    "1:\n"                                               \
    "addi     t0, t0, 8\n"                               \
    "ble      %[elemsRow], t0, 2f\n"

#define STEP2                                            \
    "addi %[actAddr], %[actAddr], 0x8\n"                 \
    "fadd.pi f29, f29, f30\n"                            \
    "beq      zero, zero, 1b\n"                          \
    "2:\n"                                               \
    "fxor.pi  f0, f0, f0\n"                              \
    "addi     t0, t0, -8\n"                              \
    "sub      t0, %[elemsRow], t0\n"                     \
    "addi     t1, zero, 1\n"                             \
    "sll      t1, t1, t0\n"                              \
    "addi     t1, t1, -1\n"                              \
    "mov.m.x  m0, t1, 0\n"                               \

#define STEP3                                            \
    "fmvs.x.ps t0, f31, 0x4\n"                           \
    "fmv.w.x   f30, t0\n"                                \
    "fadd.s    f31, f30, f31\n"                          \
    "mov.m.x m0, zero, 0x1\n"                            \
    "fbc.ps f30, 0x0(%[biasAddr])\n"                     \
    "fadd.s f31, f30, f31\n"                             \
    "fbc.ps f26, 0x8(%[offset]) \n"                      \
    "fbc.ps f27, 0x8(%[scale]) \n"                       \


template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_I8
    STEP2
    MATMUL_ITERATION_U8_I8
    STEP3
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){
  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_I8_U8
    STEP2
    MATMUL_ITERATION_I8_U8
    STEP3
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_I8_I8
    STEP2
    MATMUL_ITERATION_I8_I8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, int8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_I8_U8
    STEP2
    MATMUL_ITERATION_I8_U8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, int8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_I8
    STEP2
    MATMUL_ITERATION_U8_I8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");


}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, int8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset) {

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_U8
    STEP2
    MATMUL_ITERATION_U8_U8
    STEP3
    FP32_TO_INT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}

template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<std::is_same<src1Type, uint8_t>::value && std::is_same<src2Type, uint8_t>::value && std::is_same<dstType, uint8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){

  __asm__ __volatile__(
    STEP1
    MATMUL_ITERATION_U8_U8
    STEP2
    MATMUL_ITERATION_U8_U8
    STEP3
    FP32_TO_UINT8(f31)
    "fscb.ps f31, f28(%[dstAddr])\n"

    :
    : [gthValuesAct] "r" (gatherValuesAct),
      [gthValuesWgt] "r" (gatherValuesWgt),
      [wgtRegStep] "r" (&wgtRegStep),
      [elemsRow] "r" (elemsRow),
      [actAddr] "r" (actAddr),
      [wgtAddr] "r" (wgtAddr),
      [dstAddr] "r" (dstAddr),
      [biasAddr] "r" (biasAddr),
      [scale] "r" (scale),
      [offset] "r" (offset)
    : "t0", "t1", "f0", "f1", "f2", "f26", "f27", "f28", "f29", "f30", "f31", "memory");

}



template <typename src1Type, typename src2Type, typename dstType, typename std::enable_if<!std::is_same<src1Type, int8_t>::value && !std::is_same<src1Type, float16>::value && !std::is_same<src1Type, float>::value && !std::is_same<src1Type, uint8_t>::value, std::size_t>::type = 0>
void fullyConnectedOp (uintptr_t dstAddr, uintptr_t actAddr, uintptr_t wgtAddr, unsigned int elemsRow, int32_t gatherValuesAct[], int32_t gatherValuesWgt[], unsigned int wgtRegStep, uintptr_t biasAddr, float *scale, int32_t *offset){}

template <typename src1Type, typename src2Type, typename dstType>
void dnn_lib::fwdLibETSOCFullyConnectedInstVectorized(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches1,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    float *scale, int32_t *offset, uint64_t flags) {

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches1;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<src1Type>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead, minionId, activeMinions);
  if (maxRead == 0)
    return;

  unsigned int coord[2];
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 2, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  unsigned int offsetAIn = coord[0]*actPitch[0];

  int32_t gatherValuesAct[8], gatherValuesWgt[8];
  gatherValuesAct[0] = gatherValuesWgt[0] = 0;
  unsigned int step = weightPitch[0]*typeSize;
  for (unsigned int i = 1; i < 8; ++i) {
    gatherValuesAct[i] = gatherValuesAct[i - 1] + typeSize;
    gatherValuesWgt[i] = gatherValuesWgt[i - 1] + step;
  }
  unsigned int wgtRegStep = 8*step;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;

  while (!done && (offsetOut < posMax)) {
    uintptr_t dstAddr = (uintptr_t)dstMatrix + typeSize*offsetOut;
    uintptr_t actAddr = (uintptr_t)activations + typeSize*offsetAIn;
    uintptr_t wgtAddr = (uintptr_t)weights + typeSize*coord[1];
    uintptr_t biasAddr = (uintptr_t)bias + 4*coord[1]; // bias is a float vector.
    fullyConnectedOp <src1Type, src2Type, dstType>(dstAddr, actAddr, wgtAddr, actIndex[1], gatherValuesAct,
                       gatherValuesWgt, wgtRegStep, biasAddr, scale, offset);
    done = getOffsets(2, coord, offsetOut, dstIndex, dstPitch);
    if (coord[1] == 0) {
      offsetAIn += actPitch[0];
    }
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}
