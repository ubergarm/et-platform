
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibSparseLengthsWeightedSumInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInst<FloatTy>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSparseLengthsWeightedSumInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseLengthsWeightedSumInst"Threaded"<FloatTy>(out0, in0, in1, in2, in3, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSparseLengthsWeightedSumInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseLengthsWeightedSumInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, LibTensor* out3, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
