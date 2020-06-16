
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibSparseToDenseMaskInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseToDenseMaskInst<in0Type>(out0, in0, in1, in2, in3, Mask, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSparseToDenseMaskInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseToDenseMaskInstThreaded<in0Type>(out0, in0, in1, in2, in3, Mask, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSparseToDenseMaskInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseMaskInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, const std::array<uint64_t, default_mask_size>& Mask, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
