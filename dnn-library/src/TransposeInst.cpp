
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTransposeInst(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInst<FloatTy>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTransposeInstThreaded(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInstThreaded<FloatTy>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTransposeInstVectorized(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInstVectorized<FloatTy>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTransposeInstAligned32Bytes(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInstAligned32Bytes<FloatTy>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTransposeInst<FloatTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Float16Ty>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int8QTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int64ITy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int32ITy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int16QTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstVectorized<FloatTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstVectorized<Float16Ty>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstVectorized<Int8QTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstVectorized<Int64ITy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstVectorized<Int32ITy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstVectorized<Int16QTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<FloatTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Float16Ty>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int8QTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int64ITy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int32ITy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInstAligned32Bytes<Int16QTy>(LibTensor* out0, LibTensor* in0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
