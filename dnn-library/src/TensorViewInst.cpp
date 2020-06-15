
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTensorViewInst(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTensorViewInst<FloatTy>(in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTensorViewInstThreaded(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTensorViewInstThreaded<FloatTy>(in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTensorViewInstVectorized(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTensorViewInstVectorized<FloatTy>(in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTensorViewInst<FloatTy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Float16Ty>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int8QTy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int64ITy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int32ITy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int16QTy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstThreaded<FloatTy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstThreaded<Float16Ty>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstThreaded<Int8QTy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstThreaded<Int64ITy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstThreaded<Int32ITy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstThreaded<Int16QTy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstVectorized<FloatTy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstVectorized<Float16Ty>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstVectorized<Int8QTy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstVectorized<Int64ITy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstVectorized<Int32ITy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInstVectorized<Int16QTy>(LibTensor* in0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
