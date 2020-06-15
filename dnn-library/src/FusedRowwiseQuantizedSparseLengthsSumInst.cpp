
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<FloatTy>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Threaded"<FloatTy>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Vectorized"<FloatTy>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsSumInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
