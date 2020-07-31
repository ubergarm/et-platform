
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibConvTransposeInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibConvTransposeInst<out0Type>(out0, in0, in1, in2, Kernels, Strides, Pads, Group, Dilation, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibConvTransposeInst<FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const std::array<uint32_t, default_kernels_size> & Kernels, const std::array<uint32_t, default_kernels_size> & Strides, const std::array<uint32_t, default_kernels_size> & Pads, const uint32_t Group, const uint32_t Dilation, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
