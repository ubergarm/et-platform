
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibConvolution3DInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolution3DInst<out0Type>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibConvolution3DInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolution3DInstThreaded<out0Type>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibConvolution3DInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
