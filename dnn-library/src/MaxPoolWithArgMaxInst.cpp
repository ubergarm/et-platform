
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibMaxPoolWithArgMaxInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxPoolWithArgMaxInst<out0Type, in0Type>(out0, out1, in0, Kernels, Strides, Pads, Layout, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibMaxPoolWithArgMaxInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolWithArgMaxInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
