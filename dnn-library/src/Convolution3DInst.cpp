
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibConvolution3DInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolution3DInst<FloatTy>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibConvolution3DInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolution3DInst"Threaded"<FloatTy>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibConvolution3DInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
