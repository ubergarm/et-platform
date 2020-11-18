
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibMaxPoolInst(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxPoolInst<out0Type, in0Type>(out0, in0, Kernels, Strides, Pads, Layout, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibMaxPoolInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_pads_size> & Pads, const uint32_t Layout, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
