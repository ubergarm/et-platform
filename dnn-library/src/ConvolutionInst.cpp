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
void fwdLibConvolutionInst(void *dstMatrix, void *dstMatrixDims,
                                    void *dstMatrixPitches, void *activations,
                                    void *activationsDims,
                                    void *activationsPitches, void *weights,
                                    void *weightsDims, void *weightPitches,
                                    void *bias, void *pkernels, void *pstrides,
                                    void *ppads, unsigned int group,
                                    const float *scale, const int32_t *offset) {

  dnn_lib::inlining::fwdLibConvolutionInst<srcType>(dstMatrix, dstMatrixDims,
                                    dstMatrixPitches, activations,
                                    activationsDims,
                                    activationsPitches, weights,
                                    weightsDims, weightPitches,
                                    bias, pkernels, pstrides,
                                    ppads, group,
                                    scale, offset);
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
void fwdLibConvolutionInstThreaded(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    void *pkernels, void *pstrides, void *ppads, unsigned int group,
    const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibConvolutionInstThreaded<srcType>(
    dstMatrix, dstMatrixDims, dstMatrixPitches,
    activations, activationsDims, activationsPitches,
    weights, weightsDims, weightPitches, bias,
    pkernels, pstrides, ppads, group,
    scale, offset, flags);
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
template <typename src1Type, typename src2Type, typename dstType>
void fwdLibConvolutionInstVectorized(
    void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
    void *activations, void *activationsDims, void *activationsPitches,
    void *weights, void *weightsDims, void *weightPitches, void *bias,
    void *pkernels, void *pstrides, void *ppads, unsigned int group,
    const float *scale, const int32_t *offset, uint64_t flags) {

  dnn_lib::inlining::fwdLibConvolutionInstVectorized<src1Type, src2Type, dstType>(
    dstMatrix, dstMatrixDims, dstMatrixPitches,
    activations, activationsDims, activationsPitches,
    weights, weightsDims, weightPitches, bias,
    pkernels, pstrides, ppads, group,
    scale, offset, flags);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibConvolutionInst, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                              void *activations, void *activationsDims, void *activationsPitches,
                              void *weights, void *weightsDims, void *weightPitches, void *bias,
                              void *pkernels, void *pstrides, void *ppads, unsigned int group,
                              const float *scale, const int32_t *offset);

GEN_INSTANCES_OP(template, fwdLibConvolutionInstThreaded, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                              void *activations, void *activationsDims, void *activationsPitches,
                              void *weights, void *weightsDims, void *weightPitches, void *bias,
                              void *pkernels, void *pstrides, void *ppads, unsigned int group,
                              const float *scale, const int32_t *offset, uint64_t flags);

GEN_INSTANCES_3TYPE_OP(template, fwdLibConvolutionInstVectorized, void *dstMatrix, void *dstMatrixDims, void *dstMatrixPitches,
                              void *activations, void *activationsDims, void *activationsPitches,
                              void *weights, void *weightsDims, void *weightPitches, void *bias,
                              void *pkernels, void *pstrides, void *ppads, unsigned int group,
                              const float *scale, const int32_t *offset, uint64_t flags);

} // namespace dnn_lib
