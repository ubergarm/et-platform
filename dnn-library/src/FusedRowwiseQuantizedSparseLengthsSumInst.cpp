
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<out0Type>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsSumInstThreaded<out0Type>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsSumInstVectorized<out0Type>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstVectorized<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
