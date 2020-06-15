
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibAvgPoolInst(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibAvgPoolInst<FloatTy, FloatTy>(out0, in0, Kernels, Strides, Pads, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibAvgPoolInst"Threaded"(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibAvgPoolInst"Threaded"<FloatTy, FloatTy>(out0, in0, Kernels, Strides, Pads, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibAvgPoolInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst<Int64Ty,Int64Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst"Threaded"<FloatTy, FloatTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst"Threaded"<Float16Ty, Float16Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst"Threaded"<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst"Threaded"<UInt8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst"Threaded"<Int8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst"Threaded"<UInt8QTy,UInt8QTy>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibAvgPoolInst"Threaded"<Int64Ty,Int64Ty>(LibTensor* out0, LibTensor* out0, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
