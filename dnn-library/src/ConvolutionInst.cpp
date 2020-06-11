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

#include "ConvolutionInst.h" // From include/inlining

namespace dnn_lib {

/**
 * @brief Performs the conolution operation between the activation, weights and bias.
 *
 * This convolution admits the division of the chanel into gropus and the use of stride
 * in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
 * The convolution is executed by the first minion only.
 * 
 * @tparam elK Type of the elements of the tensors involved in the 
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
template <ElemKind dstElK, ElemKind src1ElK, ElemKind src2ElK>
void fwdLibConvolutionInst(LibTensor* outT, LibTensor* in1T, LibTensor* in2T,
                           LibTensor* in3T, void *pkernels, void *pstrides,
                           void *ppads, unsigned int group) {

  dnn_lib::inlining::fwdLibConvolutionInst<dstElk, src1Elk, src2Elk>(outT, in1T, in2T, in3T,
                                                    pkernels, pstrides, ppads,
                                                    group);
}

/**
 * @brief Performs the convolution operation between the activation, weights and bias.
 *
 * This convolution admits the division of the chanel into gropus and the use of stride
 * in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
 * This is the threaded version for the convolution.
 * 
 * @tparam elK Type of the elements of the tensors involved in the 
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
template <ElemKind dstElK, ElemKind src1ElK, ElemKind src2ElK>
void fwdLibConvolutionInstThreaded(LibTensor* outT, LibTensor* in1T,
                                   LibTensor* in2T, LibTensor* in3T,
                                   void *pkernels, void *pstrides,
                                   void *ppads, unsigned int group,
                                   uint64_t flags) {

  dnn_lib::inlining::fwdLibConvolutionInstThreaded<dstElk, src1Elk, src2Elk>(outT, in1T, in2T,
                                                            in3T, pkernels,
                                                            pstrides, ppads,
                                                            group, flags);
}

/**
 * @brief Performs the convolution operation between the activation, weights and bias.
 *
 * This convolution admits the division of the chanel into gropus and the use of stride
 * in the two dimensions of the matrix and padding to avoid loosing size of the tensor.
 * This is the threaded and vectorized version for the convolution.
 * 
 * @tparam src1Type Type of the elements of the src1 tensor involved in the 
 *  convolution (except for the bias)
 * @tparam src2Type Type of the elements of the src2 tensor involved in the 
 *  convolution (except for the bias)
 * @tparam dstType Type of the elements of the dst tensor involved in the 
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
template <ElemKind dstElK, ElemKind src1ElK, ElemKind src2ElK>
void fwdLibConvolutionInstVectorized(LibTensor* outT, LibTensor* in1T,
                                     LibTensor* in2T, LibTensor* in3T,
                                     void *pkernels, void *pstrides,
                                     void *ppads, unsigned int group,
                                     const float *scale, const int32_t *offset,
                                     uint64_t flags) {

  dnn_lib::inlining::fwdLibConvolutionInstVectorized<dstElk, src1Elk, src2Elk>(
      outT, in1T, in2T, in3T, pkernels, pstrides, ppads, group, scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_3TYPE_OP(template, fwdLibConvolutionInst, LibTensor* outT, LibTensor* in1T,
                       LibTensor* in2T, LibTensor* in3T, void *pkernels, void *pstrides,
                       void *ppads, unsigned int group);
  
GEN_INSTANCES_3TYPE_OP(template, fwdLibConvolutionInstThreaded, LibTensor* outT,
                       LibTensor* in1T, LibTensor* in2T, LibTensor* in3T, void *pkernels,
                       void *pstrides, void *ppads, unsigned int group, uint64_t flags);
  
GEN_INSTANCES_3TYPE_OP(template, fwdLibConvolutionInstVectorized, LibTensor* outT,
                       LibTensor* in1T, LibTensor* in2T, LibTensor* in3T,
                       void *pkernels, void *pstrides, void *ppads, unsigned int group,
                       const float *scale, const int32_t *offset, uint64_t flags);

} // namespace dnn_lib
