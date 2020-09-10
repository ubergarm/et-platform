
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTransposeInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInst<in0Type>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTransposeInstAligned32Bytes(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInstAligned32Bytes<in0Type>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTransposeInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, max_tensor_dimensions> & Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
