/*-------------------------------------------------------------------------
* Copyright (C) 2018, Esperanto Technologies Inc.
* The copyright to the computer program(s) herein is the
* property of Esperanto Technologies, Inc. All Rights Reserved.
* The program(s) may be used and/or copied only with
* the written permission of Esperanto Technologies and
* in accordance with the terms and conditions stipulated in the
* agreement/contract under which the program(s) have been supplied.
*-------------------------------------------------------------------------
*/

#include "LibNodes.h"

// ET insert
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

using namespace dnn_lib;

#define staticAssertFloatingPointType(ElemTy)                                  \
  static_assert(                                                               \
      std::is_floating_point<ElemTy>::value ||                                 \
          std::is_same<float16_t,                                              \
                       typename std::remove_cv<ElemTy>::type>::value,          \
      "This implementation is for floating-point values only")
/*
template <class T>
static void fwdLibETSOCMaxSplat_FloatImpl(Tensor *inW, Tensor *outW, T splatVal)
{

  staticAssertFloatingPointType(T);

  Handle<T> inHandle = inW->getHandle<T>();
  Handle<T> outHandle = outW->getHandle<T>();

  for (size_t i = 0, e = inHandle.size(); i < e; i++) {
    T val = inHandle.raw(i);
    outHandle.raw(i) = (val > splatVal) ? val : splatVal;
  }
}

template <class T>
static void fwdLibETSOCMaxSplat_I8Impl(Tensor *inW, Tensor *outW, T splatVal) {

  Handle<T> inHandle = inW->getHandle<T>();
  Handle<T> outHandle = outW->getHandle<T>();

  TensorQuantizationParams inQ{ inHandle.getType().getScale(),
                                inHandle.getType().getOffset() };
  TensorQuantizationParams outQ{ outHandle.getType().getScale(),
                                 outHandle.getType().getOffset() };
  TensorQuantizationParams splatQ{ 1.0, 0 };

  for (size_t i = 0, e = inHandle.size(); i < e; i++) {
    // Convert both sides to the destination scale and perform a regular
    // comparison.
    float val = float(inHandle.raw(i) - inQ.offset);
    float splat = float(quantization::dequantize(splatVal, splatQ));
    float relu = (val > splat) ? val : splat;
    if ((inQ.offset == -128) && (splatVal == 0))
      outHandle.raw(i) = inHandle.raw(i);
    else
      outHandle.raw(i) = std::nearbyintf(relu * (inQ.scale / outQ.scale));
  }
}

template <class T>
static void fwdLibETSOCFullyConnected_FloatImpl(Tensor *inW, Tensor *outW,
                                             Tensor *fW, Tensor *bW) {
  staticAssertFloatingPointType(T);

  Handle<T> inH = inW->getHandle<T>();
  Handle<T> outH = outW->getHandle<T>();
  Handle<T> fH = fW->getHandle<T>();
  Handle<T> bH = bW->getHandle<T>();

  auto destDim = outH.dims();
  auto srcDim = inH.dims();

  outH.clear(0);

  // For each (x,y) in the destination matrix:
  for (size_t x = 0; x < destDim[0]; x++) {
    for (size_t y = 0; y < destDim[1]; y++) {
      // Perform DOT on the row an column.
      float sum = 0;
      for (size_t i = 0; i < srcDim[1]; i++) {
        sum += float(inH.at({ x, i }) * T(1)) * float(fH.at({ i, y }) * T(1));
      }
      sum += float(bH.at({y}));
      outH.at({x,y}) = T(sum);
    }
  }
}

template <class T>
static void fwdLibETSOCFullyConnected_I8Impl(Tensor *inW, Tensor *outW, Tensor
*fW,
                                          Tensor *bW) {

  // std::cout << "FULLY CONNECTED INT8\n\n\n\n"<< std::endl;
  Handle<T> inH = inW->getHandle<T>();
  Handle<T> outH = outW->getHandle<T>();
  Handle<T> fH = fW->getHandle<T>();
  Handle<int32_t> bH = bW->getHandle<int32_t>();
  Handle<int8_t> bi8H = bW->getHandle<int8_t>();

  auto destDim = outH.dims();
  auto srcDim = inH.dims();

  float MatMulScale = inH.getType().getScale() * fH.getType().getScale();
  float scaleOut = MatMulScale / outH.getType().getScale();
  float scaleB = bH.getType().getScale() / MatMulScale;
  int32_t inOffset = inH.getType().getOffset();
  int32_t fOffset = fH.getType().getOffset();
  int32_t outOffset = outH.getType().getOffset();
  outH.clear(0);

  // For each (x,y) in the destination matrix:
  for (size_t x = 0; x < destDim[0]; x++) {
    for (size_t y = 0; y < destDim[1]; y++) {

      // Perform DOT on the row an column.
      int32_t sum = 0;
      int32_t L;
      int32_t R;
      for (size_t i = 0; i < srcDim[1]; i++) {
        if (inOffset == 0)
          L = inH.at({ x, i }); // int32 representation
        else
          L = inH.at({ x, i }) & 0xFF; // uint32 representation

        if (fOffset == 0)
          R = fH.at({ i, y }); // int32 representation
        else
          R = fH.at({ i, y }) & 0xFF; // uint32 representation

        sum += (L * R);
      }

      float bfp = bH.at({ y });
      // Scale the bias to match the scale of the matrix multiplication.
      if (bH.getType().getElementType() == ElemKind::Int8QTy) {
        bfp = float(bi8H.at({ y }));
      }
      int32_t B = std::nearbyintf(bfp * scaleB);
      sum += B;
      int32_t res = std::nearbyintf(scaleOut * sum);
      if (outOffset == 0)
        outH.at({ x, y }) = quantization::clip<int32_t, int8_t>(res);
      else
        outH.at({ x, y }) = quantization::clip<int32_t, uint8_t>(res);
    }
  }
}

void LibNodes::fwdLibETSOCMaxSplatInst(const ETSOCMaxSplatInst *I) {
  auto inW = getTensor(I->getSrc());
  auto outW = getTensor(I->getDest());

  if (inW->getType().isQuantizedType()) {
    fwdLibETSOCMaxSplat_I8Impl<int8_t>(inW, outW, I->getSplatValue());
  } else if (inW->getType().getElementType() == ElemKind::FloatTy) {
    fwdLibETSOCMaxSplat_FloatImpl<float>(inW, outW, I->getSplatValue());
  } else if (inW->getType().getElementType() == ElemKind::Float16Ty) {
    fwdLibETSOCMaxSplat_FloatImpl<float16_t>(inW, outW, I->getSplatValue());
  } else {
    llvm_unreachable("Type not supported");
  }
}

void
LibNodes::fwdLibETSOCFullyConnectedInst(const ETSOCFullyConnectedInst *I) {

  auto inW = getTensor(I->getSrc());
  auto outW = getTensor(I->getDest());
  auto fW = getTensor(I->getFilter());
  auto bW = getTensor(I->getBias());

  if (inW->getType().isQuantizedType()) {
    fwdLibETSOCFullyConnected_I8Impl<int8_t>(inW, outW, fW, bW);
  } else if (inW->getType().getElementType() == ElemKind::FloatTy) {
    fwdLibETSOCFullyConnected_FloatImpl<float>(inW, outW, fW, bW);
  } else if (inW->getType().getElementType() == ElemKind::Float16Ty) {
    fwdLibETSOCFullyConnected_FloatImpl<float16_t>(inW, outW, fW, bW);
  } else {
    llvm_unreachable("Type not supported");
  }
}
*/
