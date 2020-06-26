
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibBatchedReduceMinInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceMinInst<in0Type>(out0, in0, Axes, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibBatchedReduceMinInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_axes_size> & Axes, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
