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

/**
 * @brief Performs the conolution operation between the activation, weights and bias.
 *
 * This convolution admits the division of the chanel into gropus and the use of stride
 * in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
 * The convolution is executed by the first minion only.
 * 
 * @tparam srcType Type of the elements of the tensors involved in the 
 *  convolution (except for the bias)
 * @param[out] dstMatrix Matrix in wich we save the result of the convolution.
 * @param[in] dstMatrixDims Vector of dimensions of the dstMatrix 
 *  (with batch and chanel).
 * @param[in] dstMatrixPitches Vector of pitches of the dstMatrix.
 * @param[in] weights Matrix with the weights for the convolution.
 * @param[in] weightDims Vector of dimensions of the weights. Unused.
 * @param[in] weightPitches Vector of pitches of the weights.
 * @param[in] bias Floats vector of biases (one for each chanel in a group).
 * @param[in] pkernels Vector of dimensions of the kernek that is applied.
 * @param[in] pstrides Vector with the strides for both dimensions.
 * @param[in] ppads Vector with the padding for both dimensions.
 * @param[in] group The number of groups in which we divide the chanel.
 * @param[in] scale The scale for the quantization.
 * @param[in] offset The offset for the quantization.
 */
template <typename srcType>
void dnn_lib::fwdLibConvolutionInst(void *dstMatrix, void *dstMatrixDims,
                                    void *dstMatrixPitches, void *activations,
                                    void *activationsDims,
                                    void *activationsPitches, void *weights,
                                    void *weightsDims, void *weightPitches,
                                    void *bias, void *pkernels, void *pstrides,
                                    void *ppads, unsigned int group,
                                    float *scale, int32_t *offset) {

  // FIXME: going back to single thread until general case is solved with
  // multithread
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

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides; 
  unsigned int *pads = (unsigned int *)ppads; 

  assert(actIndex[3] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 &&
         "Output channels must be divisible by group.");
  size_t inCperG = actIndex[3] / group;
  size_t outCperG = dstIndex[3] / group;

  // For each input in the batch:
  for (size_t n = 0; n < actIndex[0]; n++) {

    // For each group of input channels:
    for (size_t g = 0; g < group; g++) {

      // For each output channel in the group:
      for (size_t d = g * outCperG; d < (g + 1) * outCperG; d++) {

        // For each convolution 'jump' in the input tensor:
        ssize_t x = -ssize_t(pads[0]);
        for (size_t ax = 0; ax < dstIndex[1]; x += strides[0], ax++) {
          ssize_t y = -ssize_t(pads[1]);
          for (size_t ay = 0; ay < dstIndex[2]; y += strides[1], ay++) {

            // For each element in the convolution-filter:
            auto sum = tAInput[0];
            sum = 0;
            for (size_t fx = 0; fx < kernels[0]; fx++) {
              for (size_t fy = 0; fy < kernels[1]; fy++) {
                ssize_t ox = x + fx;
                ssize_t oy = y + fy;

                // Ignore index access below zero (this is due to padding). The
                // elegance of this should be improved
                if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
                    oy >= ssize_t(actIndex[2])) {
                  continue;
                }

                for (size_t fd = 0; fd < inCperG; fd++) {
                  auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                                     fy * weightPitch[2] + fd];
                  auto op2 =
                      tAInput[n * actPitch[0] + (size_t)ox * actPitch[1] +
                              (size_t)oy * actPitch[2] + g * inCperG + fd];
                  sum += op1 * op2;
                }
              }
            }
            int64_t addr = n * dstPitch[0] + (size_t)ax * dstPitch[1] +
                           (size_t)ay * dstPitch[2] + d;
            sum += tBias[d];
            tOutput[addr] = sum;
          } // W
        }   // H
      }     // C
    }       // G
  }         // N
}

/**
 * @brief Performs the convolution operation between the activation, weights and bias.
 *
 * This convolution admits the division of the chanel into gropus and the use of stride
 * in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
 * This is the threaded version for the convolution.
 * 
 * @tparam srcType Type of the elements of the tensors involved in the 
 *  convolution (except for the bias)
 * @param[out] dstMatrix Matrix in wich we save the result of the convolution.
 * @param[in] dstMatrixDims Vector of dimensions of the dstMatrix 
 *  (with batch and chanel).
 * @param[in] dstMatrixPitches Vector of pitches of the dstMatrix.
 * @param[in] weights Matrix with the weights for the convolution.
 * @param[in] weightDims Vector of dimensions of the weights. Unused.
 * @param[in] weightPitches Vector of pitches of the weights.
 * @param[in] bias Floats vector of biases (one for each chanel in a group).
 * @param[in] pkernels Vector of dimensions of the kernek that is applied.
 * @param[in] pstrides Vector with the strides for both dimensions.
 * @param[in] ppads Vector with the padding for both dimensions.
 * @param[in] group The number of groups in which we divide the chanel.
 * @param[in] scale The scale for the quantization.
 * @param[in] offset The offset for the quantization.
 * @param[in] flags Controls the active shires and the type of evict that 
 *  should be done at the end of the function.
 */
template <typename srcType>
void dnn_lib::fwdLibConvolutionInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    void *pkernels, void *pstrides, void *ppads, unsigned int group,
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

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides; 
  unsigned int *pads = (unsigned int *)ppads; 


  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  assert(actIndex[3] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 &&
         "Output channels must be divisible by group.");
  unsigned int inCperG = actIndex[3] / group;
  unsigned int outCperG = dstIndex[3] / group;

  unsigned int eDstPitch[5] = {dstPitch[0], dstPitch[1], dstPitch[2], outCperG,
                               1};

  unsigned int eDstIndex[5] = {dstIndex[0], dstIndex[1], dstIndex[2], group,
                               outCperG};

  unsigned int coord[5], k;
  getNonPaddingCoordinates(coord, initialAddr, 5, eDstPitch, eDstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, d;
  while ((offsetOut < posMax) && !done) {
    x = coord[1] * strides[0] - ssize_t(pads[0]); // least x coordinate in kernel
    y = coord[2] * strides[1] - ssize_t(pads[1]); // least y coordinate in kernel
    d = coord[3] * outCperG + coord[4];           // depth in kernel

    auto sum = tAInput[0];                        //Same type as tAInput[]
    sum = tBias[d];// 0;

    for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }
    //sum += tBias[d];
    tOutput[offsetOut] = sum;

    done = getOffsets(5, coord, offsetOut, eDstIndex, eDstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}


/**
 * @brief Computes one element in the convolution.
 *
 * This consists on the vectorized implementation for the products of 
 * convolutionInst, which works computing the product of the elements 
 * in the filter with the activations in groups of up to 8 elements and 
 * sums them together at the end.
 * 
 * @tparam srcType The type of the elements in the matrix, which in this funcion is
 *  is imposed to be float.
 * @param[in] activations Matrix of activations for the convolution.
 * @param[in] weights Matrix of weights for the convolution.
 * @param[in] coord The vector of coordinates to the initial position in the 
 *  activations. coord[0] corresponds to the batch and coord[3] corresponds 
 *  to the group where we are.
 * @param[in] actPitch Vector of pitches of the activations matrix.
 * @param[in] weightPitch Vector of pitches of the weights matrix.
 * @param[in] actIndex Vector of the size of each dimensions of the activations.
 * @param[in] kernels Dimensions of the filters or kernels.
 * @param[in] inCperG Elements in a group.
 * @param[out] sum The result of applying the filter in the given position.
 * @param[in] mask The int32_t that determines which lanes should be active when 
 *  we can't take 8 elements at the same time.
 * @param[in] x, y, d Coordinates where our minions should start reading.
 */
template <typename srcType, typename std::enable_if<std::is_same<
                            srcType, float>::value, std::size_t>::type = 0>
void convolutionOp (void *activations, void *weights, unsigned int *coord,
                    unsigned int *actPitch, unsigned int *weightPitch,
                    unsigned int *actIndex, unsigned int *kernels,
                    unsigned int inCperG, float &sum, int32_t mask, ssize_t x,
                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
  int dist;
  ssize_t fx, fy, ox, oy;
  fx = fy = 0;
  unsigned int *actAddr = (unsigned int *) activations;
  unsigned int *weightAddr = (unsigned int *) weights;
  actAddr += coord[0] * actPitch[0] + x * actPitch[1] + y * actPitch[2] +
            coord[3] * inCperG;
  weightAddr += d * weightPitch[0];
  __asm__ __volatile__(
    "fxor.pi  f0, f0, f0\n"                         // f0 to zeros
    "mov.m.x  m0, zero, 0xff\n"                     // m0 to ones
    "mov.m.x  m1, %[mask], 0\n"                     // m1 the auxiliar mask
    "1:\n"                                          // for (size_t fx = 0; fx < kernels[0]; fx++) {
    "beq      %[fy], zero, 2f\n"
    "mul      %[fy], %[kernels1], %[actPitch2]\n"
    "sub      %[actAddr], %[actAddr], %[fy]\n"
    "mul      %[fy], %[kernels1], %[weightPitch2]\n"
    "sub      %[weightAddr], %[weightAddr], %[fy]\n"
    "addi     %[fy], zero, 0\n"
    "2:\n"                                            // for (size_t fy = 0; fy < kernels[1]; fy++) {
    "addi     %[dist], %[inCperG], 0\n"                // dist = inCperG

    "add      %[oy], %[y], %[fy]\n"                     // oy = y + fy
    "add      %[ox], %[x], %[fx]\n"                     // ox = x + fx

    "blt      %[ox], zero, 5f\n"                        // if (ox < 0) continue
    "blt      %[oy], zero, 5f\n"                        // if (oy < 0) continue
    "ble      %[actIndex1], %[ox], 5f\n"                // if (actIndex[1] <= ox) continue
    "ble      %[actIndex2], %[oy], 5f\n"                // if (actIndex[2] <= oy) continue

    "addi     t0, zero, 8\n"                            // t0 = 8
    "ble      %[dist], t0, 4f\n"                        // if dist <= 8 go to 4

    "mov.m.x  m0, zero, 0xff\n"
    "3:\n"                                              // while (8 < dist) {
    "flw.ps   f1, 0x0(%[actAddr])\n"                      // actAddr -> f1
    "flw.ps   f2, 0x0(%[weightAddr])\n"                   // weightaddr -> f2
    "fmadd.ps f0, f1, f2, f0\n"                           // f0 = (f1 * f2) + f0
    "addi     %[actAddr], %[actAddr], 32\n"               // actAddr += 32
    "addi     %[weightAddr], %[weightAddr], 32\n"         // weightAddr += 32
    "addi     %[dist], %[dist], -8\n"                     // dist -= 8
    "blt      t0, %[dist], 3b\n"                        // }

    "4:\n"
    "maskand  m0, m0, m1\n"                             // put mask on
    "flw.ps   f1, 0x0(%[actAddr])\n"                    // actAddr -> f1
    "flw.ps   f2, 0x0(%[weightAddr])\n"                 // weightaddr -> f2
    "fmadd.ps f0, f1, f2, f0\n"                         // f0 = (f1 * f2) + f0
    "sub      %[dist], %[inCperG], %[dist]\n"           // dist = inCperG - dist
    "slli     %[dist], %[dist], 2\n"                    // dist = dist * 4
    "sub      %[actAddr], %[actAddr], %[dist]\n"        // actAddr = actAddr - dist
    "sub      %[weightAddr], %[weightAddr], %[dist]\n"  // actAddr = actAddr - dist

    "5:\n"
    "addi     %[fy], %[fy], 1\n"                        // fy++
    "add     %[actAddr], %[actPitch2], %[actAddr]\n"   // actAddr = actAddr + actPitch[2]
    "add     %[weightAddr], %[weightPitch2], %[weightAddr]\n"
    "blt      %[fy], %[kernels1], 2b\n"               // Closing fy for }

    "addi     %[fx], %[fx], 1\n"                      // fx++

    "add     %[actAddr], %[actPitch1], %[actAddr]\n" // actAddr = actAddr + actPitch[1]
    "add     %[weightAddr], %[weightPitch1], %[weightAddr]\n"
    "blt      %[fx], %[kernels0], 1b\n"             // Closing fx for{}

    "mov.m.x   m0, zero, 0xff\n"
    "fswizz.ps f1, f0, 0xe\n"
    "fadd.ps   f0, f0, f1\n"
    "fswizz.ps f1, f0, 0x1\n"
    "fadd.ps   f0, f0, f1\n"
    "fmvs.x.ps t0, f0, 0x4\n"
    "fmv.w.x   f31, t0\n"
    "fadd.s    f31, f31, f0\n"

    "fmv.w.x   f0, %[sum]\n"
    "fadd.s    f31, f31, f0\n"
    "fmv.x.w   %[sum], f31\n"

    : [ weightAddr ] "+r" (weightAddr),
      [ actAddr ] "+r" (actAddr),
      [ dist ] "+r" (dist),
      [ sum ] "+r" (sum),
      [ ox ] "+r" (ox),
      [ oy ] "+r" (oy),
      [ fy ] "+r" (fy),
      [ fx ] "+r" (fx)
    : [ weightPitch1 ] "r" (weightPitch[1] * 4),
      [ weightPitch2 ] "r" (weightPitch[2] * 4),
      [ actIndex1 ] "r" (actIndex[1]),
      [ actIndex2 ] "r" (actIndex[2]),
      [ actPitch1 ] "r" (actPitch[1] * 4),
      [ actPitch2 ] "r" (actPitch[2] * 4),
      [ kernels0 ] "r" (kernels[0]),
      [ kernels1 ] "r" (kernels[1]),
      [ inCperG ] "r" (inCperG),
      [ mask ] "r" (mask),
      [ x ] "r" (x),
      [ y ] "r" (y)
    : "memory", "f0", "f1", "f2", "f31", "t0", "t1");
  return;
}

/**
 * @brief Computes one element in the convolution.
 *
 * @overload
 */
template <typename srcType, typename std::enable_if<std::is_same<
                            srcType, float16>::value, std::size_t>::type = 0>
void convolutionOp (void *activations, void *weights, unsigned int *coord,
                    unsigned int *actPitch, unsigned int *weightPitch,
                    unsigned int *actIndex, unsigned int *kernels,
                    unsigned int inCperG, float16 &sum, int32_t mask, ssize_t x,
                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
  int dist;
  ssize_t fx, fy, ox, oy;
  fx = fy = 0;
  uint16_t *actAddr = (uint16_t *) activations;
  uint16_t *weightAddr = (uint16_t *) weights;
  actAddr += coord[0] * actPitch[0] + x * actPitch[1] + y * actPitch[2] +
            coord[3] * inCperG;
  weightAddr += d * weightPitch[0];
  unsigned int gatherValues[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };
  __asm__ __volatile__(
    "flw.ps f16, 0x0(%[gatherValues])\n"
    "fxor.pi  f0, f0, f0\n"                         // f0 to zeros
    "mov.m.x  m0, zero, 0xff\n"                     // m0 to ones
    "mov.m.x  m1, %[mask], 0\n"                     // m1 the auxiliar mask
    "1:\n"                                          // for (size_t fx = 0; fx < kernels[0]; fx++) {
    "beq      %[fy], zero, 2f\n"
    "mul      %[fy], %[kernels1], %[actPitch2]\n"
    "sub      %[actAddr], %[actAddr], %[fy]\n"
    "mul      %[fy], %[kernels1], %[weightPitch2]\n"
    "sub      %[weightAddr], %[weightAddr], %[fy]\n"
    "addi     %[fy], zero, 0\n"
    "2:\n"                                            // for (size_t fy = 0; fy < kernels[1]; fy++) {
    "addi     %[dist], %[inCperG], 0\n"                // dist = inCperG

    "add      %[oy], %[y], %[fy]\n"                     // oy = y + fy
    "add      %[ox], %[x], %[fx]\n"                     // ox = x + fx

    "blt      %[ox], zero, 5f\n"                        // if (ox < 0) continue
    "blt      %[oy], zero, 5f\n"                        // if (oy < 0) continue
    "ble      %[actIndex1], %[ox], 5f\n"                // if (actIndex[1] <= ox) continue
    "ble      %[actIndex2], %[oy], 5f\n"                // if (actIndex[2] <= oy) continue

    "addi     t0, zero, 8\n"                            // t0 = 8
    "ble      %[dist], t0, 4f\n"                        // if dist <= 8 go to 4

    "mov.m.x  m0, zero, 0xff\n"
    "3:\n"                                              // while (8 < dist) {
    "fgh.ps   f1, f16(%[actAddr])\n"                      // actAddr -> f1
    "fcvt.ps.f16 f1, f1\n"
    "fgh.ps   f2, f16(%[weightAddr])\n"                   // weightaddr -> f2
    "fcvt.ps.f16 f2, f2\n"
    "fmadd.ps f0, f1, f2, f0\n"                           // f0 = (f1 * f2) + f0
    "addi     %[actAddr], %[actAddr], 16\n"               // actAddr += 16
    "addi     %[weightAddr], %[weightAddr], 16\n"         // weightAddr += 16
    "addi     %[dist], %[dist], -8\n"                     // dist -= 8
    "blt      t0, %[dist], 3b\n"                        // }

    "4:\n"
    "maskand  m0, m0, m1\n"                             // put mask on
    "fgh.ps   f1, f16(%[actAddr])\n"                    // actAddr -> f1
    "fcvt.ps.f16 f1, f1\n"
    "fgh.ps   f2, f16(%[weightAddr])\n"                 // weightaddr -> f2
    "fcvt.ps.f16 f2, f2\n"
    "fmadd.ps f0, f1, f2, f0\n"                         // f0 = (f1 * f2) + f0
    "sub      %[dist], %[inCperG], %[dist]\n"           // dist = inCperG - dist
    "slli     %[dist], %[dist], 1\n"                    // dist = dist * 2
    "sub      %[actAddr], %[actAddr], %[dist]\n"        // actAddr = actAddr - dist
    "sub      %[weightAddr], %[weightAddr], %[dist]\n"  // actAddr = actAddr - dist

    "5:\n"
    "addi     %[fy], %[fy], 1\n"                        // fy++
    "add      %[actAddr], %[actPitch2], %[actAddr]\n"   // actAddr = actAddr + actPitch[2]
    "add      %[weightAddr], %[weightPitch2], %[weightAddr]\n"
    "blt      %[fy], %[kernels1], 2b\n"               // Closing fy for{}

    "addi     %[fx], %[fx], 1\n"                      // fx++

    "add      %[actAddr], %[actPitch1], %[actAddr]\n" // actAddr = actAddr + actPitch[1]
    "add      %[weightAddr], %[weightPitch1], %[weightAddr]\n"
    "blt      %[fx], %[kernels0], 1b\n"             // Closing fx for{}

    "mov.m.x   m0, zero, 0xff\n"
    "fswizz.ps f1, f0, 0xe\n"
    "fadd.ps   f0, f0, f1\n"
    "fswizz.ps f1, f0, 0x1\n"
    "fadd.ps   f0, f0, f1\n"
    "fmvs.x.ps t0, f0, 0x4\n"
    "fmv.w.x   f31, t0\n"
    "fadd.s    f31, f31, f0\n"
    "fmv.w.x   f0, %[sum]\n"
    "fadd.s    f31, f31, f0\n"
    "fmv.x.w   %[sum], f31\n"

    : [ weightAddr ] "+r" (weightAddr),
      [ actAddr ] "+r" (actAddr),
      [ dist ] "+r" (dist),
      [ sum ] "+r" (sum),
      [ ox ] "+r" (ox),
      [ oy ] "+r" (oy),
      [ fy ] "+r" (fy),
      [ fx ] "+r" (fx)
    : [ weightPitch1 ] "r" (weightPitch[1] * 2),
      [ weightPitch2 ] "r" (weightPitch[2] * 2),
      [ gatherValues ] "r" (gatherValues),
      [ actPitch1 ] "r" (actPitch[1] * 2),
      [ actPitch2 ] "r" (actPitch[2] * 2),
      [ actIndex1 ] "r" (actIndex[1]),
      [ actIndex2 ] "r" (actIndex[2]),
      [ kernels0 ] "r" (kernels[0]),
      [ kernels1 ] "r" (kernels[1]),
      [ inCperG ] "r" (inCperG),
      [ mask ] "r" (mask),
      [ x ] "r" (x),
      [ y ] "r" (y)
    : "memory", "f0", "f1", "f2", "f31", "t0", "t1");
  return;
}

/**
 * @brief Computes one element in the convolution.
 *
 * This consists on the non-vectorized implementation for the products of 
 * convolutionInst, which is the same as in the threaded version, but works
 * for all the non supported types in the vectorized version of this same 
 * function.
 * 
 * @tparam srcType The type of the elements in the matrix.
 * @param[in] activations Matrix of activations for the convolution.
 * @param[in] weights Matrix of weights for the convolution.
 * @param[in] coord The vector of coordinates to the initial position in the 
 *  activations. coord[0] corresponds to the batch and coord[3] corresponds 
 *  to the group where we are.
 * @param[in] actPitch Vector of pitches of the activations matrix.
 * @param[in] weightPitch Vector of pitches of the weights matrix.
 * @param[in] actIndex Vector of the size of each dimensions of the activations.
 * @param[in] kernels Dimensions of the filters or kernels.
 * @param[in] inCperG Elements in a group.
 * @param[out] sum The result of applying the filter in the given position.
 * @param[in] mask It has no relevance in this function.
 * @param[in] x, y, d Coordinates where our minions should start reading.
 */
template <typename srcType, typename std::enable_if<(!std::is_same<
                            srcType, float>::value) /*&& (!std::is_same<
                            srcType, float16>::value) && (!std::is_same<
                            srcType, int8_t>::value)*/, std::size_t>::type = 0>
void convolutionOp (void *activations, void *weights, unsigned int *coord,
                    unsigned int *actPitch, unsigned int *weightPitch,
                    unsigned int *actIndex, unsigned int *kernels,
                    unsigned int inCperG, float &sum, int32_t mask, ssize_t x,
                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }
  return; //TODO return error.
}

/**
 * @brief Computes one element in the convolution.
 *
 * @overload
 */
template <typename srcType, typename std::enable_if</*(!std::is_same<
                            srcType, float>::value) && */(!std::is_same<
                            srcType, float16>::value) /*&& (!std::is_same<
                            srcType, int8_t>::value)*/, std::size_t>::type = 0>
void convolutionOp (void *activations, void *weights, unsigned int *coord,
                    unsigned int *actPitch, unsigned int *weightPitch,
                    unsigned int *actIndex, unsigned int *kernels,
                    unsigned int inCperG, float16 &sum, int32_t mask, ssize_t x,
                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }
  return; //TODO return error.
}


template <typename srcType>
void convolutionOp (void *activations, void *weights, unsigned int *coord,
                    unsigned int *actPitch, unsigned int *weightPitch,
                    unsigned int *actIndex, unsigned int *kernels,
                    unsigned int inCperG, int32_t &sum, int32_t mask, ssize_t x,
                    ssize_t y, ssize_t d, float *scale, int32_t *offset) {
  const Addresser<srcType> tAInput(activations, scale[0], offset[0]);
  const Addresser<srcType> tWInput(weights, scale[1], offset[1]);
  for (size_t fx = 0; fx < kernels[0]; fx++) {  //for all x coordinates in kernel
      for (size_t fy = 0; fy < kernels[1]; fy++) {//for all y coordinates in kernel
        ssize_t ox = x + fx;
        ssize_t oy = y + fy;

        // Ignore index access below zero (this is due to padding).
        if (ox < 0 || oy < 0 || ox >= ssize_t(actIndex[1]) ||
            oy >= ssize_t(actIndex[2])) {
          continue;
        }
        for (size_t fd = 0; fd < inCperG; fd++) { //for all depth coordinates
          auto op1 = tWInput[d * weightPitch[0] + fx * weightPitch[1] +
                             fy * weightPitch[2] + fd];
          auto op2 =
              tAInput[coord[0] * actPitch[0] + (size_t)ox * actPitch[1] +
                      (size_t)oy * actPitch[2] + coord[3] * inCperG + fd];
          sum += op1 * op2;
        }
      }
    }
  return; //TODO return error.
}


/**
 * @brief Performs the convolution operation between the activation, weights and bias.
 *
 * This convolution admits the division of the chanel into gropus and the use of stride
 * in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
 * This is the threaded and vectorized version for the convolution.
 * 
 * @tparam srcType Type of the elements of the tensors involved in the 
 *  convolution (except for the bias)
 * @param[out] dstMatrix Matrix in wich we save the result of the convolution.
 * @param[in] dstMatrixDims Vector of dimensions of the dstMatrix 
 *  (with batch and chanel).
 * @param[in] dstMatrixPitches Vector of pitches of the dstMatrix.
 * @param[in] weights Matrix with the weights for the convolution.
 * @param[in] weightDims Vector of dimensions of the weights. Unused.
 * @param[in] weightPitches Vector of pitches of the weights.
 * @param[in] bias Floats vector of biases (one for each chanel in a group).
 * @param[in] pkernels Vector of dimensions of the kernek that is applied.
 * @param[in] pstrides Vector with the strides for both dimensions.
 * @param[in] ppads Vector with the padding for both dimensions.
 * @param[in] group The number of groups in which we divide the chanel.
 * @param[in] scale The scale for the quantization.
 * @param[in] offset The offset for the quantization.
 * @param[in] flags Controls the active shires and the type of evict that 
 *  should be done at the end of the function.
 */
template <typename srcType>
void dnn_lib::fwdLibConvolutionInstVectorized(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    void *pkernels, void *pstrides, void *ppads, unsigned int group,
    float *scale, int32_t *offset, uint64_t flags) {

  Addresser<srcType> tOutput(dstMatrix, scale[3], offset[3]);

  unsigned int minionId = get_minion_id();
  unsigned int activeMinions = 32 * ACTIVE_SHIRES;
  if (minionId >= activeMinions)
    return;
  float *tBias = (float *)bias;

  unsigned int *dstIndex = (unsigned int *)dstMatrixDims;
  unsigned int *actIndex = (unsigned int *)activationsDims;
  unsigned int *weightIndex = (unsigned int *)weightsDims;

  unsigned int *dstPitch = (unsigned int *)dstMatrixPitches;
  unsigned int *actPitch = (unsigned int *)activationsPitches;
  unsigned int *weightPitch = (unsigned int *)weightPitches;

  unsigned int *kernels = (unsigned int *)pkernels;
  unsigned int *strides = (unsigned int *)pstrides; // Jump between convols
  unsigned int *pads = (unsigned int *)ppads; // 0 added to avoid loss of dims


  unsigned int numElemsDst = dstPitch[0] * dstIndex[0];
  unsigned int initialAddr, maxRead;
  size_t typeSize = getsize<srcType>();
  getCachelinePartition(typeSize, numElemsDst, initialAddr, maxRead,
                        minionId, activeMinions);
  if (maxRead == 0)
    return;

  assert(actIndex[3] % group == 0 &&
         "Input channels must be divisible by group.");
  assert(dstIndex[3] % group == 0 &&
         "Output channels must be divisible by group.");
  unsigned int inCperG = actIndex[3] / group;
  unsigned int outCperG = dstIndex[3] / group;

  unsigned int eDstPitch[5] = {dstPitch[0], dstPitch[1], dstPitch[2], outCperG,
                               1};

  unsigned int eDstIndex[5] = {dstIndex[0], dstIndex[1], dstIndex[2], group,
                               outCperG};

  unsigned int coord[5], k;
  getNonPaddingCoordinates(coord, initialAddr, 5, eDstPitch, eDstIndex, k);

  unsigned int offsetOut = 0;
  for (int i = 0; i < k; i++) {
    offsetOut += coord[i] * eDstPitch[i];
  }
  if (offsetOut >= numElemsDst)
    return;

  unsigned int posMax = initialAddr + maxRead;
  bool done = false;
  ssize_t x, y, d;
  volatile int32_t mask = (1 << (((inCperG - 1) & 0x7)  + 1)) - 1;
  while ((offsetOut < posMax) && !done) {
    x = coord[1] * strides[0] - ssize_t(pads[0]);
    y = coord[2] * strides[1] - ssize_t(pads[1]);
    d = coord[3] * outCperG + coord[4];

    auto sum = tBias[d];
    volatile int dist;
    volatile unsigned int *actAddr = (unsigned int *) activations;
    volatile unsigned int *weightAddr = (unsigned int *) weights;
    convolutionOp <srcType> (activations, weights, coord, actPitch, weightPitch,
                             actIndex, kernels, inCperG, sum, mask, x, y, d,
                             scale, offset);
    tOutput[offsetOut] = sum;

    done = getOffsets(5, coord, offsetOut, eDstIndex, eDstPitch);
  }
  if (!DO_EVICTS)
    return;
  unsigned int clperminion = maxRead * typeSize / 64;
  if (clperminion > 0) evict_va_multi(DO_EVICTS, (uintptr_t)dstMatrix + typeSize*initialAddr, clperminion);
}

GEN_INSTANCES_OP(template, fwdLibConvolutionInst, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                              void *activations, void *activationsDims, void *activationsPitches,
                              void *weights, void *weightsDims, void *weightPitches, void *bias,
                              void *pkernels, void *pstrides, void *ppads, unsigned int group,
                              float *scale, int32_t *offset);
GEN_INSTANCES_OP(template, fwdLibConvolutionInstThreaded, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                              void *activations, void *activationsDims, void *activationsPitches,
                              void *weights, void *weightsDims, void *weightPitches, void *bias,
                              void *pkernels, void *pstrides, void *ppads, unsigned int group,
                              float *scale, int32_t *offset, uint64_t flags);
GEN_INSTANCES_OP(template, fwdLibConvolutionInstVectorized, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                              void *activations, void *activationsDims, void *activationsPitches,
                              void *weights, void *weightsDims, void *weightPitches, void *bias,
                              void *pkernels, void *pstrides, void *ppads, unsigned int group,
                              float *scale, int32_t *offset, uint64_t flags);
