
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibExtractTensorInst(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibExtractTensorInst<FloatTy>(out0, in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibExtractTensorInst"Threaded"(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibExtractTensorInst"Threaded"<FloatTy>(out0, in0, Offsets, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibExtractTensorInst<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibExtractTensorInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
