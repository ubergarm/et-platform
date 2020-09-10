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

#ifndef _AVG_POOL_INST_H_
#define _AVG_POOL_INST_H_

#include <assert.h>
#include <fenv.h>
#include <limits>
#include <cmath>
#include <cstring>

#include "Float16.h"
#include "Writer.h" // From include/internal path
#include "Addresser.h" // From include/internal path
#include "Converter.h" // From include/internal path
#include "Operator.h" // From include/internal path
#include "utils.h" // From include/internal path

#include "LibTypes.h"
#include "LibUtils.h"
#include "LibCommon.h"
#include "LibTensor.h"


namespace dnn_lib {

namespace inlining {


template <ElemKind dstElK, size_t N, size_t PN>
inline typename std::enable_if_t<(isQuantizedElemKind(dstElK) || (dstElK == Float16Ty)), void> 
 fwdLibAvgPool3DInst(LibTensor* outT, LibTensor* inT,
         const std::array<uint32_t, N> &kernels,
         const std::array<uint32_t, N> &strides,
         const std::array<uint32_t, PN> &pads,
         uint64_t flags,
         const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;
  
  assert(inT->getElementType() == outT->getElementType());
  assert((inT->getElementType() == Float16Ty)||(inT->getElementType() == Int8QTy));

  using dstType = typename elemKind2elemTy<dstElK>::type;

  auto inH = inT->getHandle<dstType>();
  auto outH = outT->getHandle<dstType>();

  float invFilterArea = 0.0;
  fpReciprocalSingleElement((kernels[0] * kernels[1] * kernels[2]), invFilterArea);

  //For each input in the batch
  for (size_t n = 0; n < outT->dims()[0]; n++) {
    //For each layer in the output tensor:
    for (size_t z = 0; z < inT->dims()[4]; z++) {
      //For each convolution 'jump' in the input tensor:
      ssize_t t = -size_t(pads[0]);
      for (size_t at = 0; at < outT->dims()[1]; t += strides[0], at++) {
  ssize_t x = -ssize_t(pads[2]);
  for (size_t ax = 0; ax < outT->dims()[2]; x += strides[1], ax++) {
    ssize_t y = -ssize_t(pads[4]);
    for (size_t ay = 0; ay < outT->dims()[3]; y += strides[2], ay++) {

      float sum = 0;
      for (size_t ft = 0; ft < kernels[0]; ft++) {
        for (size_t fx = 0; fx < kernels[1]; fx++) {
    for (size_t fy = 0; fy < kernels[2]; fy++) {
      sdim_t ot = t + ft;
      sdim_t ox = x + fx;
      sdim_t oy = y + fy;

      //Ignore index access below zero(this is due to padding).
      if (ot < 0 || ox < 0 || oy < 0 || ot >= static_cast<ssize_t>(inT->dims()[1]) ||
          ox >= static_cast<ssize_t>(inT->dims()[2]) || oy >= static_cast<ssize_t>(inT->dims()[3])) {
        continue;
      }

      if (dstElK == Float16Ty) {
        float dst = 0;
        /*the cast avoid compilation error due to quantize types are handle together here.*/
        convertFp16ToFp32(static_cast<uint16_t>(inH.at(std::array<size_t, 5>{n, 
            static_cast<dim_t>(ot),static_cast<dim_t>(ox), static_cast<dim_t>(oy), z})), dst);
        sum += dst;
      }
      else {
        sum += dequantize<dstType>(inH.at(std::array<size_t, 5>{n, static_cast<dim_t>(ot), 
          static_cast<dim_t>(ox), static_cast<dim_t>(oy), z}), inH.getScale(), inH.getOffset());
      }
    }     
        }

        if (dstElK == Float16Ty) {
    uint16_t dst = 0;
    convertFp32ToFp16((sum * invFilterArea), dst);
    outH.at(std::array<size_t,5>{n,at,ax,ay,z}) = dst;
        }
        else {
    outH.at(std::array<size_t, 5>{n,at,ax,ay,z}) = quantize<dstType>((sum * invFilterArea),
                    outH.getScale(),
                    outH.getOffset());
        }

      }
    } //W
  }   //H
      }     //T
    }       //C
  }         //N     

}

template <ElemKind dstElK, size_t N, size_t PN>
inline typename std::enable_if_t<(!isQuantizedElemKind(dstElK) && (dstElK != Float16Ty)), void> 
 fwdLibAvgPool3DInst(LibTensor* outT, LibTensor* inT,
        const std::array<uint32_t, N> &kernels,
        const std::array<uint32_t, N> &strides,
        const std::array<uint32_t, PN> &pads,
        uint64_t flags,
        const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;
  
  assert(inT->getElementType() == outT->getElementType());
  assert(inT->getElementType() == FloatTy);

  using dstType = typename elemKind2elemTy<dstElK>::type;

  auto inH = inT->getHandle<dstType>();
  auto outH = outT->getHandle<dstType>();

  float invFilterArea = 0.0;
  fpReciprocalSingleElement((kernels[0] * kernels[1] * kernels[2]), invFilterArea);

  //For each input in the batch
  for (size_t n = 0; n < outT->dims()[0]; n++) {
    //For each layer in the output tensor:
    for (size_t z = 0; z < inT->dims()[4]; z++) {
      //For each convolution 'jump' in the input tensor:
      ssize_t t = -size_t(pads[0]);
      for (size_t at = 0; at < outT->dims()[1]; t += strides[0], at++) {
  ssize_t x = -ssize_t(pads[2]);
  for (size_t ax = 0; ax < outT->dims()[2]; x += strides[1], ax++) {
    ssize_t y = -ssize_t(pads[4]);
    for (size_t ay = 0; ay < outT->dims()[3]; y += strides[2], ay++) {

      float sum = 0;
      for (size_t ft = 0; ft < kernels[0]; ft++) {
        for (size_t fx = 0; fx < kernels[1]; fx++) {
    for (size_t fy = 0; fy < kernels[2]; fy++) {
      sdim_t ot = t + ft;
      sdim_t ox = x + fx;
      sdim_t oy = y + fy;

      //Ignore index access below zero(this is due to padding).
      if (ot < 0 || ox < 0 || oy < 0 || ot >= static_cast<ssize_t>(inT->dims()[1]) ||
          ox >= static_cast<ssize_t>(inT->dims()[2]) || oy >= static_cast<ssize_t>(inT->dims()[3])) {
        continue;
      }

      sum += static_cast<float>(inH.at(std::array<size_t, 5>{n, static_cast<dim_t>(ot), static_cast<dim_t>(ox), static_cast<dim_t>(oy), z}));
    }
        }
        outH.at(std::array<size_t, 5>{n,at,ax,ay,z}) = (sum * invFilterArea);
      }
    } //W
  }   //H
      }     //T
    }       //C
  }         //N     
}



  template <ElemKind dstElK, size_t N, size_t PN>
inline  typename std::enable_if_t<(isQuantizedElemKind(dstElK) || (dstElK == Float16Ty)), void>
 fwdLibAvgPoolInst(LibTensor* outT, LibTensor* inT,
       const std::array<uint32_t, N> &kernels,
       const std::array<uint32_t, N> &strides,
       const std::array<uint32_t, PN> &pads,
       uint64_t flags, 
       const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(inT->getElementType() == outT->getElementType());
  assert((inT->getElementType() == Float16Ty)||(inT->getElementType() == Int8QTy));
  
  if (inT->ndims() == 5) {
    fwdLibAvgPool3DInst<dstElK, N, PN>(outT, inT, kernels, strides, pads, flags, 
               minionOffset, assignedMinions);
    return;
  }

  using dstType = typename elemKind2elemTy<dstElK>::type;

  auto inH = inT->getHandle<dstType>();
  auto outH = outT->getHandle<dstType>();

  float invFilterArea = 0.0;
  fpReciprocalSingleElement((kernels[0] * kernels[1]), invFilterArea);

  // For each input in the batch:
  for (size_t n = 0; n < outT->dims()[0]; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < inT->dims()[3]; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pads[0]);
      for (size_t ax = 0; ax < outT->dims()[1]; x += strides[0], ax++) {
        ssize_t y = -ssize_t(pads[1]);
        for (size_t ay = 0; ay < outT->dims()[2]; y += strides[1], ay++) {

          float sum = 0;
          for (size_t fx = 0; fx < kernels[0]; fx++) {
            for (size_t fy = 0; fy < kernels[1]; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(inT->dims()[1]) ||
                  oy >= ssize_t(inT->dims()[2])) {
                continue;
              }

        if (dstElK == Float16Ty) {
    float dst = 0;
    /*the cast avoid compilation error due to quantize types are handle together here.*/
    convertFp16ToFp32(static_cast<uint16_t>(inH.at(std::array<size_t, 4>{n, 
        static_cast<dim_t>(ox), static_cast<dim_t>(oy), z})), dst);
    sum += dst;
        }
        else {
    sum += dequantize<dstType>(inH.at(std::array<size_t, 4>{n, 
      static_cast<dim_t>(ox), static_cast<dim_t>(oy), z}), inH.getScale(), inH.getOffset());
        }      
            }
          }

    if (dstElK == Float16Ty) {
      uint16_t dst = 0;
      convertFp32ToFp16((sum * invFilterArea), dst);
      outH.at(std::array<size_t,4>{n,ax,ay,z}) = dst;
    }
    else {
      outH.at(std::array<size_t, 4>{n,ax,ay,z}) = quantize<dstType>((sum * invFilterArea),
                    outH.getScale(),
                    outH.getOffset());
    }
        } // W
      }   // H
    }     // C
  }       // N
}


  template <ElemKind dstElK, size_t N, size_t PN>
inline  typename std::enable_if_t<(!isQuantizedElemKind(dstElK) && (dstElK != Float16Ty)), void>
 fwdLibAvgPoolInst(LibTensor* outT, LibTensor* inT,
       const std::array<uint32_t, N> &kernels,
       const std::array<uint32_t, N> &strides,
       const std::array<uint32_t, PN> &pads,
       uint64_t flags, 
       const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {

  if (get_minion_id() != minionOffset) return;

  assert(inT->getElementType() == outT->getElementType());
  assert((inT->getElementType() == Float16Ty)||(inT->getElementType() == Int8QTy));
  
  if (inT->ndims() == 5) {
    fwdLibAvgPool3DInst<dstElK, N, PN>(outT, inT, kernels, strides, pads, flags, 
               minionOffset, assignedMinions);
    return;
  }

  using dstType = typename elemKind2elemTy<dstElK>::type;

  auto inH = inT->getHandle<dstType>();
  auto outH = outT->getHandle<dstType>();

  float invFilterArea = 0.0;
  fpReciprocalSingleElement((kernels[0] * kernels[1]), invFilterArea);

  // For each input in the batch:
  for (size_t n = 0; n < outT->dims()[0]; n++) {
    // For each layer in the output tensor:
    for (size_t z = 0; z < inT->dims()[3]; z++) {
      // For each convolution 'jump' in the input tensor:
      ssize_t x = -ssize_t(pads[0]);
      for (size_t ax = 0; ax < outT->dims()[1]; x += strides[0], ax++) {
        ssize_t y = -ssize_t(pads[1]);
        for (size_t ay = 0; ay < outT->dims()[2]; y += strides[1], ay++) {

          float sum = 0;
          for (size_t fx = 0; fx < kernels[0]; fx++) {
            for (size_t fy = 0; fy < kernels[1]; fy++) {
              ssize_t ox = x + fx;
              ssize_t oy = y + fy;

              // Ignore index access below zero (this is due to padding).
              if (ox < 0 || oy < 0 || ox >= ssize_t(inT->dims()[1]) ||
                  oy >= ssize_t(inT->dims()[2])) {
                continue;
              }
        sum += inH.at(std::array<size_t, 4>{n, static_cast<dim_t>(ox), static_cast<dim_t>(oy), z});
            }
          }
    outH.at(std::array<size_t, 4>{n,ax,ay,z}) = (sum * invFilterArea);
  
  } // W
      }   // H
    }     // C
  }       // N
}

  template <ElemKind dstElK, size_t N, size_t PN>
inline void fwdLibAvgPoolInstThreaded(LibTensor* outT, LibTensor* inT,
                                      const std::array<uint32_t, N> &kernels,
                                      const std::array<uint32_t, N> &strides,
                                      const std::array<uint32_t, PN> &pads,
                                      uint64_t flags,
                                      const uint32_t minionOffset = 0, const uint32_t assignedMinions = 0) {
  using dstType = typename elemKind2elemTy<dstElK>::type;

  unsigned int minionId = get_minion_id() - minionOffset;
  unsigned int activeMinions = (assignedMinions == 0) ? (MIN_PER_SHIRE * ACTIVE_SHIRES) : assignedMinions;
  if (minionId >= activeMinions) return;

  if (inT->ndims() == 5) {
    fwdLibAvgPool3DInst<dstElK, N, PN>(outT, inT, kernels, strides, pads, flags, 
               minionOffset, assignedMinions);
    return;
  }

  void* src = inT->getRawDataPointer<void>();
  void* dst = outT->getRawDataPointer<void>();
  
  // Addresser<dstElK> tOutput(dstMatrix, scale[1], offset[1]);
  Addresser<dstElK> tOutput(dst, outT->getScale(), outT->getOffset());
  // const Addresser<srcElK> tAInput(activations, scale[0], offset[0]);
  const Addresser<dstElK> tAInput(src, inT->getScale(), outT->getOffset());
 
  // unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  const dim_t *dstIndex = outT->dims().data();
  // unsigned int *actIndex = (unsigned int *)activationsDims;
  const dim_t *actIndex = inT->dims().data();
  // unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  const dim_t *dstPitch = outT->strides().data();
  // unsigned int *actPitch = (unsigned int *)activationsPitches;
  const dim_t *actPitch = inT->strides().data();
  
  float filterArea = kernels[0] * kernels[1];
  float invFilter;
  fpReciprocalSingleElement(filterArea, invFilter);

  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<dstType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions, dst);
  if (maxRead == 0)
    return;

  unsigned int coord[4] = {0, 0, 0, 0};
  unsigned int k = 0;
  getNonPaddingCoordinates(coord, initialAddr, 4, dstPitch, dstIndex, k);

  unsigned int offsetOut = 0;
  for (unsigned int i = 0; i < k; i++) {
    offsetOut += coord[i] * dstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y;
  while (!done && (offsetOut < posMax)) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);

    auto sum = tAInput[0];
    sum = 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {
      for (size_t fy = 0; fy < kernels[1]; fy++) {
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }

        sum += tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                       (size_t)oy * actPitch[2] + coord[3] * actPitch[3]];
      }
    }

    float tmp = sum;
    tmp *= invFilter;
    sum = tmp;
    tOutput[offsetOut] = sum;

    done = getOffsets(4, coord, offsetOut, dstIndex, dstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / CACHE_LINE_BYTES;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dst + typeSize*initialAddr, clperminion);
}

} // namespace inlining

} // namespace dnn_lib

#endif // _AVG_POOL_INST_H_
