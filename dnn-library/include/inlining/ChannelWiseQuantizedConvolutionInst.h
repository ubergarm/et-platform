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

#ifndef _CHANNEL_WISE_QUANTIZED_CONVOLUTION_INST_H_
#define _CHANNEL_WISE_QUANTIZED_CONVOLUTION_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>
#include <math.h>

#include "Float16.h"
#include "utils.h" // From include/internal path
#include "LibTypes.h"
#include "LibTensor.h"
#include "LibCommon.h"

namespace dnn_lib {

namespace inlining {

/**
 * @brief 
 *
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @tparam Elemkind the kind of the element which hast to be resolved.
 * @param[out] outT LibTensor destination. It holds the expected data.
 * @param[in] dataT LibTensor input. It keeps the Data.
 * @param[in] filterT LibTensor input. It keeps filter data.
 * @param[in] biasT LibTensor input. It keeps bias data.
 * @param[in] fsT LibTensor input. It keeps filtes scale data.
 * @param[in] foT LibTensor input. It keeps filter offset data.
 * @param[in] bsT LibTensor input. It keeps bias scale data.
 * @param[in] boT LibTensor input. It keeps bias offster data
 * @param[in] kernels
 * @param[in] strides
 * @param[in] pads
 * @param[in] group
 * @param[in] dilation
 * @param[flags] flags Gives the information of the Active Shires and the
 *  type of evict required.
 */
template <ElemKind dstElK, ElemKind src2ElK, size_t N, size_t PN, size_t KN>
inline void fwdLibChannelWiseQuantizedConvolution3DInst(LibTensor* outT, LibTensor* dataT, 
            LibTensor* filterT, LibTensor* biasT,
            LibTensor* fsT, LibTensor* foT,
            LibTensor* bsT, LibTensor* boT,
            const std::array<uint32_t, N> &kernels,
            const std::array<uint32_t, N> &strides,
            const std::array<uint32_t, PN> &pads,
            const uint32_t group, 
            const std::array<uint32_t, KN> &dilation,
            const uint64_t flags, 
            const uint32_t minionOffset = 0, 
            const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(dstElK == Int8QTy);
  assert(dataT->getElementType() == filterT->getElementType());
  assert(outT->getElementType() == dataT->getElementType());

  assert((src2ElK == Int8QTy) || (src2ElK == Int32QTy));

  /* ob #samples, oh height, ow witdh, oc #channels*/
  /* data and output channel must be divisible by group. */
  assert((dataT->dims()[4] % group)==0);
  assert((outT->dims()[4] % group)==0);

  size_t inCperG = (dataT->dims()[4] / group);
  size_t outCperG = (outT->dims()[4] / group);

  using elkType = typename elemKind2elemTy<dstElK>::type;
  using biasType = typename elemKind2elemTy<src2ElK>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = dataT->getHandle<elkType>();
  auto filterH = filterT->getHandle<elkType>();
  auto biasH = biasT->getHandle<biasType>();
  auto fsH = fsT->getHandle<float>();
  auto foH = foT->getHandle<int32_t>();
  auto bsH = bsT->getHandle<float>();
  auto boH = boT->getHandle<int32_t>();

  float inScale = dataH.getScale();
  int32_t inOffset = dataH.getOffset();
  /* float invOutScale = 0.0; */
  /* int32_t outOffset = outH.getOffset(); */
  /* fpReciprocalSingleElement(outH.getScale(), invOutScale); */

  // for each input in the batch
  for (size_t n = 0; n < dataT->dims()[0]; n++) {
    // for each group of input channels
    for (size_t g = 0; g < group; g++) {
      // for each output channel in the group
      for (size_t d = (g * outCperG); d < ((g + 1) * outCperG); d++) {

  //Get channel wise quantization params.
  int32_t filterOffset = foH.at(std::array<size_t,1>{d});
  float filterScale = fsH.at(std::array<size_t,1>{d});
  int32_t biasOffset = boH.at(std::array<size_t,1>{d});
  float biasScale = bsH.at(std::array<size_t,1>{d});
  float matMulScale = (inScale * filterScale);
  float invMatMulScale = 0.0;
  fpReciprocalSingleElement(matMulScale, invMatMulScale);

  // For each convolution 'jump' in the input tensor:
  ssize_t t = -ssize_t(pads[0]); //near
  for (size_t at = 0; at < outT->dims()[1]; t += strides[0], at++) {
    ssize_t x = -ssize_t(pads[2]); //top
    for (size_t ax = 0; ax < outT->dims()[2]; x += strides[1], ax++) {
      ssize_t y = -ssize_t(pads[4]); //left
      for (size_t ay = 0; ay < outT->dims()[3]; y += strides[2], ay++) {

        //For each element in the convolution-filter:
        int32_t sum = 0;
        for (size_t ft = 0; ft < kernels[0]; ft++) {
    for (size_t fx = 0; fx < kernels[1]; fx++) { 
      for (size_t fy = 0; fy < kernels[2]; fy++) {
        ssize_t ot = t + ft;
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        //Ignore index access below zero (this is due to padding).
        if (ot < 0 || ox < 0 || oy < 0 || ot >= ssize_t(dataT->dims()[1]) ||
      ox >= ssize_t(dataT->dims()[2]) || oy >= ssize_t(dataT->dims()[3])) {
          continue;
        }

        //Accumulate along the filter depth
        for (size_t fd = 0; fd < inCperG; fd++) {
          biasType F = filterH.at(std::array<size_t, 5>{d, ft, fx ,fy, fd});
          biasType I = dataH.at(std::array<size_t, 5>{n, size_t(ot), size_t(ox),
          size_t(oy), (g * inCperG + fd)});

          //We represent the element multiplication with offset as (value-offset).
          sum += (F - filterOffset) * (I - inOffset);
        }
      }
    }
        }

        // Scale the bias to match the scale of the matrix mulitplication.
        sum += nearbyintf(static_cast<float>(biasH.at(std::array<size_t, 1>{d}) - biasOffset) *
        (invMatMulScale * biasScale));

        //Scale the result back to the expected destination scale.
        outH.at(std::array<size_t,5>{n, at, ax, ay, d}) = 
    quantize<elkType>(sum * matMulScale, outT->getScale(), outT->getOffset());

      } // W
    }   // H
  }     // T
      }       // C
    }         // G
  }           // N
}

template <ElemKind dstElK, ElemKind src2ElK, size_t N, size_t PN, size_t KN>
inline void fwdLibChannelWiseQuantizedConvolutionInst(LibTensor* outT, LibTensor* dataT, 
            LibTensor* filterT, LibTensor* biasT,
            LibTensor* fsT, LibTensor* foT,
            LibTensor* bsT, LibTensor* boT,
            const std::array<uint32_t, N> &kernels,
            const std::array<uint32_t, N> &strides,
            const std::array<uint32_t, PN> &pads,
            const uint32_t group, 
            const std::array<uint32_t, KN> &dilation,
            const uint64_t flags, 
            const uint32_t minionOffset = 0, 
            const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(dstElK == Int8QTy);
  assert(dataT->getElementType() == filterT->getElementType());
  assert(outT->getElementType() == dataT->getElementType());

  assert((src2ElK == Int8QTy) || (src2ElK == Int32QTy));

  /* ob #samples, oh height, ow witdh, oc #channels*/
  /* data and output channel must be divisible by group. */
  assert((dataT->dims()[3] % group)==0);
  assert((outT->dims()[3] % group)==0);

  if (dataT->ndims() == 5) {
    fwdLibChannelWiseQuantizedConvolution3DInst<dstElK, src2ElK>(outT, dataT,
      filterT, biasT, fsT, foT, bsT, boT, kernels, strides, pads, 
                  group, dilation, flags, minionOffset, assignedMinions);
    return;
  }

  size_t inCperG = (dataT->dims()[3] / group);
  size_t outCperG = (outT->dims()[3] / group);

  using elkType = typename elemKind2elemTy<dstElK>::type;
  using biasType = typename elemKind2elemTy<src2ElK>::type;

  auto outH = outT->getHandle<elkType>();
  auto dataH = dataT->getHandle<elkType>();
  auto filterH = filterT->getHandle<elkType>();
  auto biasH = biasT->getHandle<biasType>();
  auto fsH = fsT->getHandle<float>();
  auto foH = foT->getHandle<int32_t>();
  auto bsH = bsT->getHandle<float>();
  auto boH = boT->getHandle<int32_t>();

  outH.zero();
 

  float inScale = dataH.getScale();
  int32_t inOffset = dataH.getOffset();
  /* float invOutScale = 0.0; */
  /* int32_t outOffset = outH.getOffset(); */
  /* fpReciprocalSingleElement(outH.getScale(), invOutScale); */

  // for each group of input channels
  for (size_t n = 0; n < dataT->dims()[0]; n++) {
    // for each group of input channels
    for (size_t g = 0; g < group; g++) {
      // for each output channel in the group
      for (size_t d = (g * outCperG); d < ((g + 1) * outCperG); d++) {

  //Get channel wise quantization params.
  int32_t filterOffset = foH.at(std::array<size_t,1>{d});
  float filterScale = fsH.at(std::array<size_t,1>{d});
  int32_t biasOffset = boH.at(std::array<size_t,1>{d});
  float biasScale = bsH.at(std::array<size_t,1>{d});
  float matMulScale = (inScale * filterScale);
  float invMatMulScale = 0.0;
  fpReciprocalSingleElement(matMulScale, invMatMulScale);

  // For each convolution 'jump' in the input tensor:
  ssize_t x = -ssize_t(pads[0]);
  for (size_t ax = 0; ax < outT->dims()[1]; x += strides[0], ax++) {
    ssize_t y = -ssize_t(pads[1]);
    for (size_t ay = 0; ay < outT->dims()[2]; y += strides[1], ay++) {

      //For each element in the convolution filter:
      int32_t sum = 0;
      for (size_t fx = 0; fx < kernels[0]; fx++) {
        for (size_t fy = 0; fy < kernels[1]; fy++) {
    ssize_t ox = x + fx * dilation[0];
    ssize_t oy = y + fy * dilation[1];
    
    //Ignore index access below zero (this is due to padding)
    if (ox < 0 || oy < 0 || ox >= ssize_t(dataT->dims()[1]) ||
        oy >= ssize_t(dataT->dims()[2])) {
      continue;
    }

    //Accumulate along the filter depth.

    for (size_t fd = 0; fd < inCperG; fd++) {
      biasType F = filterH.at(std::array<size_t,4>{d, fx, fy, fd});
      biasType I = dataH.at(std::array<size_t,4>{n, static_cast<size_t>(ox), static_cast<size_t>(oy), (g * inCperG + fd)});
      //We represent the element multiplication with offset as (value - offset)
      sum += (F - filterOffset) * ( I - inOffset);
    }
        }
      }
      
      // Scale the bias to match the scale of the matrix multiplication.
      sum += nearbyintf(static_cast<float>(biasH.at(std::array<size_t,1>{d}) - biasOffset) *
            (invMatMulScale * biasScale));

      printf("mdbg_ Scale bias sum %d\tmatMulScale=%f\toutScale=%f\toutOffset=%d\n",sum,matMulScale, outT->getScale(), outT->getOffset());

      // Scale the bias to match the scale of the matrix mulitplication.
      outH.at(std::array<size_t,4>{n, ax, ay, d}) = 
        quantize<elkType>(sum * matMulScale, outT->getScale(), outT->getOffset());

      printf("[%lu,%lu,%lu,%lu] =  %d\n",n,ax,ay,d,static_cast<elkType>(outH.at(std::array<size_t,4>{n,ax,ay,d})));

    } // W
  }   // H
      }     // C
    }       // G
  }         // N

    
}

}
}
#endif
