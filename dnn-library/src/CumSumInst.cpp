
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibCumSumInst(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibCumSumInst<in0Type>(out0, in0, Exclusive, Reverse, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibCumSumInst<FloatTy>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCumSumInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCumSumInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibCumSumInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const bool Exclusive, const bool Reverse, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
