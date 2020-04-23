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

#include "CopyInst.h" // From include/inlining

namespace dnn_lib {

/**
 * @brief Copies the src matrix to the dst matrix.
 *
 * It makes a copy of the tensor src into the dst tensor, which may not
 * have the same pitches or dimensions, so it allows a reshaping. In this
 * version all the work is done by the same minion.
 * 
 * @tparam srcType The type of the elements in the tensor.
 * @param[out] dst Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] src Pointer to the input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] srcPitches Vector of pitches of the input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] scale, offset Parameters for the quantization.
 */
template <typename srcType>
void fwdLibCopyInst(LibTensor* outT, LibTensor* inT) {

  dnn_lib::inlining::fwdLibCopyInst<srcType>(outT, inT);
}

/**
 * @brief Copies the src matrix to the dst matrix.
 *
 * It makes a copy of the tensor src into the dst tensor, which may not
 * have the same pitches or dimensions, so it allows a reshaping. This is
 * the threaded version for this operator, so several minions are used.
 * 
 * @tparam srcType The type of the elements in the tensor.
 * @param[out] dst Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] src Pointer to the input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] srcPitches Vector of pitches of the input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] scale, offset Parameters for the quantization.
 * @param[in] flags Gives the information of the Active Shires and the
 *  type of evict required.
 * @param[in] minionOffset The first minion that is assigned to this node.
 * @param[in] assignedMinions Amount of minions avaliable.
 */
template <typename srcType>
void fwdLibCopyInstThreaded(LibTensor* outT, LibTensor* inT,
                            uint64_t flags,
                            const uint32_t minionOffset,
                            const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibCopyInstThreaded<srcType>(outT, inT, flags, minionOffset,
                                                     assignedMinions);
}

/**
 * @brief Copies the src matrix to the dst matrix.
 *
 * It makes a copy of the tensor src into the dst tensor, which may not
 * have the same pitches or dimensions, so it allows a reshaping. This is
 * the threaded and vectorized version for this operator.
 *
 * @Warning It is assumed that the destination tensor starts at the beginning
 *  of a cacheline.
 * 
 * @tparam srcType The type of the elements in the tensor.
 * @param[out] dst Pointer to the output matrix.
 * @param[in] dstDims The "number of dimensions" of the output matrix.
 * @param[in] dstPitches Vector of pitches of the output matrix.
 * @param[in] src Pointer to the input matrix.
 * @param[in] srcDims The vector of dimensions of the input tensor.
 * @param[in] srcPitches Vector of pitches of the input tensor.
 * @param[in] srcDimNum The "number of dimensions" of the input matrix.
 * @param[in] scale, offset Parameters for the quantization.
 * @param[in] flags Gives the information of the Active Shires and the
 *  type of evict required.
 * @param[in] minionOffset The first minion that is assigned to this node.
 * @param[in] assignedMinions Amount of minions avaliable.
 */
template <typename srcType>
void fwdLibCopyInstVectorized(LibTensor* outT, LibTensor* inT, uint64_t flags,
                              const uint32_t minionOffset,
                              const uint32_t assignedMinions) {

  dnn_lib::inlining::fwdLibCopyInstVectorized<srcType>(outT, inT, flags,
                                                       minionOffset, assignedMinions);
}

#include "GenInstances.h"

GEN_INSTANCES_OP(template, fwdLibCopyInst, LibTensor* outT, LibTensor* inT);

GEN_INSTANCES_OP(template, fwdLibCopyInstThreaded, LibTensor* outT, LibTensor* inT,
                 uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);

GEN_INSTANCES_OP(template, fwdLibCopyInstVectorized, LibTensor* outT, LibTensor* inT,
                 uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
  
} // namespace dnn_lib
