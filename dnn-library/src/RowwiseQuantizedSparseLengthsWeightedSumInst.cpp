
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in4Type>
  void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<out0Type, in4Type>(out0, in0, in1, in2, in3, in4, in5, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in4Type>
  void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstThreaded<out0Type, in4Type>(out0, in0, in1, in2, in3, in4, in5, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in4Type>
  void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<out0Type, in4Type>(out0, in0, in1, in2, in3, in4, in5, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstThreaded<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstThreaded<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstThreaded<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstThreaded<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibRowwiseQuantizedSparseLengthsWeightedSumInstVectorized<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
