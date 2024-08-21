/*-------------------------------------------------------------------------
 * Copyright (C) 2021, Esperanto Technologies Inc.
 * The copyright to the computer program(s) herein is the
 * property of Esperanto Technologies, Inc. All Rights Reserved.
 * The program(s) may be used and/or copied only with
 * the written permission of Esperanto Technologies and
 * in accordance with the terms and conditions stipulated in the
 * agreement/contract under which the program(s) have been supplied.
 *-------------------------------------------------------------------------
 */

#ifndef _LIB_API_IMPL_SEL_H_
#define _LIB_API_IMPL_SEL_H_

// Local
#include "LibTensor.h"

// STD
#include <vector>

namespace dnn_lib {

////////////////////////////////////////////////////////////////////////////////
// default implementation selector
////////////////////////////////////////////////////////////////////////////////
class implSel {
public:
  template <int implCount> static size_t defaultSel(std::vector<LibTensor*>&, std::vector<LibTensor*>&) {
    return implCount - 1;
  }

  ////////////////////////////////////////////////////////////////////////////////
  // custom implementation selectors
  ////////////////////////////////////////////////////////////////////////////////

  // Best implementation selector for operator ResizeNearest. Return values are:
  //   0: base implementation
  //   1: implementation for integer upscales
  static size_t ResizeNearest(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator Copy. Return values are:
  //   0: base implementation (Vectorized)
  //   1: Tensorized
  static size_t Copy(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator ElementBool instructions. Return values are:
  //   0: base implementation (threaded)
  //   1: Vectorized
  static size_t ElementBool(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  static size_t ElementCmpEQ(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  static size_t ElementCmpNEQ(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  static size_t ElementCmpLTE(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  static size_t ElementCmpLT(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator MaxSplat. Return values are:
  //   0: base implementation (vectorized)
  //   1: Aligned32Bytes
  static size_t MaxSplat(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator RowwiseQuantizedFullyConnected. Return values are:
  //   0: base implementation (Vectorized)
  //   1: Aligned32Bytes
  static size_t RowwiseQuantizedFullyConnected(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator RowwiseQuantizedSparseLengthsWeightedSum. Return values are:
  //   0: base implementation
  //   1: Vectorized
  static size_t RowwiseQuantizedSparseLengthsWeightedSum(std::vector<LibTensor*>& outTensors,
                                                         std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator Transpose. Return values are:
  //   0: base implementation (Vectorized)
  //   1: Aligned32Bytes
  static size_t Transpose(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator SoftMax. Return values are:
  //   0: base implementation
  //   1: Vectorized
  static size_t SoftMax(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator LocalResponseNormalization. Return values are:
  //   0: base implementation (threaded)
  //   1: Vectorized
  static size_t LocalResponseNormalization(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator LocalResponseNormalization. Return values are:
  //   0: base implementation (threaded)
  //   1: Threaded
  static size_t SparseLengthsSum(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator SparseLengthsWeightedSum. Return values are:
  //   0: base implementation
  //   1: Threaded
  static size_t SparseLengthsWeightedSum(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator InsertTensor. Return values are:
  //   0: base implementation
  //   1: Threaded
  static size_t InsertTensor(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator AvgPool. Return values are:
  //   0: base implementation
  //   1: Threaded
  static size_t AvgPool(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);

  // Best implementation selector for operator EmbeddingBag. Return values are:
  //   0: base implementation
  //   1: Threaded
  static size_t EmbeddingBag(std::vector<LibTensor*>& outTensors, std::vector<LibTensor*>& inTensors);
};

} // namespace dnn_lib

#endif
