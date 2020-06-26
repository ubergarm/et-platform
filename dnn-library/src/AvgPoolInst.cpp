
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibAvgPoolInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibAvgPoolInst<out0Type, in0Type>(out0, in0, Kernels, Strides, Pads, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibAvgPoolInstThreaded(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibAvgPoolInstThreaded<out0Type, in0Type>(out0, in0, Kernels, Strides, Pads, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibAvgPoolInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInstThreaded<FloatTy, FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInstThreaded<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInstThreaded<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInstThreaded<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInstThreaded<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInstThreaded<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInstThreaded<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
