
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy,UInt8QTy>(out0, in0, in1, in2, in3, in4, in5, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Threaded"<FloatTy,UInt8QTy>(out0, in0, in1, in2, in3, in4, in5, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Vectorized"<FloatTy,UInt8QTy>(out0, in0, in1, in2, in3, in4, in5, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Threaded"<FloatTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Threaded"<Float16Ty,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Vectorized"<FloatTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst"Vectorized"<Float16Ty,UInt8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, LibTensor* out4, LibTensor* out5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
