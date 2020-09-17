
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibCrossEntropyLossInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibCrossEntropyLossInst<in0Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibCrossEntropyLossInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCrossEntropyLossInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCrossEntropyLossInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCrossEntropyLossInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCrossEntropyLossInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCrossEntropyLossInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
