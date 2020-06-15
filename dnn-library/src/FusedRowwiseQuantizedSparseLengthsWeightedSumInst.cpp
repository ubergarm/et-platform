
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTy"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTy"<FloatTy>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyThreaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyThreaded"<FloatTy>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"<FloatTy>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTy"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTy"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyThreaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyThreaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFusedRowwiseQuantizedSparseLengthsWeightedSumInst"FloatTyVectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
