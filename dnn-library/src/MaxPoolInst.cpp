
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibMaxPoolInst(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxPoolInst<FloatTy, FloatTy>(out0, in0, Kernels, Strides, Pads, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibMaxPoolInst"Threaded"(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibMaxPoolInst"Threaded"<FloatTy, FloatTy>(out0, in0, Kernels, Strides, Pads, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibMaxPoolInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst<Int64Ty,Int64Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst"Threaded"<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst"Threaded"<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst"Threaded"<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst"Threaded"<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst"Threaded"<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst"Threaded"<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibMaxPoolInst"Threaded"<Int64Ty,Int64Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
