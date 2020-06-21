
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibSpaceToDepthInst(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSpaceToDepthInst<out0Type>(out0, in0, BlockSize, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSpaceToDepthInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSpaceToDepthInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSpaceToDepthInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSpaceToDepthInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint32_t BlockSize, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
