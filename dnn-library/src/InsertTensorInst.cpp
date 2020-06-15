
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibInsertTensorInst(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibInsertTensorInst<FloatTy>(out0, in0, Offsets, Count, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibInsertTensorInst"Threaded"(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibInsertTensorInst"Threaded"<FloatTy>(out0, in0, Offsets, Count, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibInsertTensorInst<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Offsets, uint32_t Count, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
