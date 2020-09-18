
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTopKInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTopKInst<in0Type>(out0, out1, in0, TopK, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTopKInst<FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTopKInst<Int16QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const uint32_t TopK, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
