
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibConvolutionInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolutionInst<FloatTy,FloatTy,FloatTy>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibConvolutionInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolutionInst"Threaded"<FloatTy,FloatTy,FloatTy>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibConvolutionInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvolutionInst"Vectorized"<FloatTy,FloatTy,FloatTy>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibConvolutionInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst<Int16QTy,Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst"Threaded"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst"Threaded"<Int16QTy,Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst"Vectorized"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolutionInst"Vectorized"<Int16QTy,Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, std::array<uint32_t, default_kernels_size> Kernels, std::array<uint32_t, default_kernels_size> Strides, std::array<uint32_t, default_kernels_size> Pads, uint32_t Group, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
