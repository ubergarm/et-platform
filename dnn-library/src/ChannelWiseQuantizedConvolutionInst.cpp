
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////

template <ElemKind out0Type, ElemKind in2Type>
void fwdLibChannelWiseQuantizedConvolutionInst(
  LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5,
  LibTensor* in6, const std::array<uint32_t, default_kernels_size>& Kernels,
  const std::array<uint32_t, default_kernels_size>& Strides, const std::array<uint32_t, default_pads_size>& Pads,
  const uint32_t Group, const std::array<uint32_t, default_dilation_size>& Dilation, const size_t FusedActivation,
  const std::array<float, default_fusedActivationArgs>& FusedActivationArgs, const uint64_t flags,
  const uint32_t minionOffset, const uint32_t assignedMinions) {
  dnn_lib::inlining::fwdLibChannelWiseQuantizedConvolutionInst<out0Type, in2Type>(
    out0, in0, in1, in2, in3, in4, in5, in6, Kernels, Strides, Pads, Group, Dilation, FusedActivation,
    FusedActivationArgs, flags, minionOffset, assignedMinions);
}

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibChannelWiseQuantizedConvolutionInst<Int8QTy, Int8QTy>(
  LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5,
  LibTensor* in6, const std::array<uint32_t, default_kernels_size>& Kernels,
  const std::array<uint32_t, default_kernels_size>& Strides, const std::array<uint32_t, default_pads_size>& Pads,
  const uint32_t Group, const std::array<uint32_t, default_dilation_size>& Dilation, const size_t FusedActivation,
  const std::array<float, default_fusedActivationArgs>& FusedActivationArgs, const uint64_t flags,
  const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibChannelWiseQuantizedConvolutionInst<Int8QTy, Int32QTy>(
  LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, LibTensor* in3, LibTensor* in4, LibTensor* in5,
  LibTensor* in6, const std::array<uint32_t, default_kernels_size>& Kernels,
  const std::array<uint32_t, default_kernels_size>& Strides, const std::array<uint32_t, default_pads_size>& Pads,
  const uint32_t Group, const std::array<uint32_t, default_dilation_size>& Dilation, const size_t FusedActivation,
  const std::array<float, default_fusedActivationArgs>& FusedActivationArgs, const uint64_t flags,
  const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
