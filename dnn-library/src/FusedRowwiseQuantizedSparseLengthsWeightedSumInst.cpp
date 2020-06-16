
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<out0Type>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstThreaded<out0Type>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<out0Type>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
