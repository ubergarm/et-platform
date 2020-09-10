
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in2Type>
  void fwdLibConvolutionInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolutionInst<out0Type, in2Type>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, Dilation, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in2Type>
  void fwdLibConvolutionInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolutionInstVectorized<out0Type, in2Type>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, Dilation, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibConvolutionInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInstVectorized<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInstVectorized<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInstVectorized<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInstVectorized<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInstVectorized<Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInstVectorized<Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
