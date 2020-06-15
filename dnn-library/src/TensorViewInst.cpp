
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTensorViewInst(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTensorViewInst<FloatTy>(in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTensorViewInst"Threaded"(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTensorViewInst"Threaded"<FloatTy>(in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTensorViewInst"Vectorized"(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTensorViewInst"Vectorized"<FloatTy>(in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTensorViewInst<FloatTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Float16Ty>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int8QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int64ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int32ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst<Int16QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Threaded"<FloatTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Threaded"<Float16Ty>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Threaded"<Int8QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Threaded"<Int64ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Threaded"<Int32ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Threaded"<Int16QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Vectorized"<FloatTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Vectorized"<Float16Ty>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Vectorized"<Int8QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Vectorized"<Int64ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Vectorized"<Int32ITy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTensorViewInst"Vectorized"<Int16QTy>(LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
