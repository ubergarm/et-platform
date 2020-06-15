
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibTransposeInst(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInst<FloatTy>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTransposeInst"Threaded"(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInst"Threaded"<FloatTy>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTransposeInst"Vectorized"(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInst"Vectorized"<FloatTy>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibTransposeInst"Aligned32Bytes"(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibTransposeInst"Aligned32Bytes"<FloatTy>(out0, in0, Shuffle, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibTransposeInst<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Vectorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Vectorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Vectorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Aligned32Bytes"<FloatTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Aligned32Bytes"<Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Aligned32Bytes"<Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Aligned32Bytes"<Int64ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Aligned32Bytes"<Int32ITy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibTransposeInst"Aligned32Bytes"<Int16QTy>(LibTensor* out0, LibTensor* out0, std::array<size_t, max_tensor_dimensions> Shuffle, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
