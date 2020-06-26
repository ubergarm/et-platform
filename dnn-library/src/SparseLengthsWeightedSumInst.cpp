
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in2Type>
  void fwdLibSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInst<in0Type, in2Type>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSparseLengthsWeightedSumInst<FloatTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Float16Ty, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int8QTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int64ITy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int32ITy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int16QTy, Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
