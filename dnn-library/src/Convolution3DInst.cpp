
#include "LibNodes.h"

namespace dnn_lib {
////////////////////////////////////////////////////////////////////////////////
// Forward call to corresponding dnn_lib::inlining implementations
////////////////////////////////////////////////////////////////////////////////

template <ElemKind out0Type, ElemKind in2Type>
void fwdLibConvolution3DInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2,
                             const std::array<uint32_t, default_kernels_size>& Kernels,
                             const std::array<uint32_t, default_kernels_size>& Strides,
                             const std::array<uint32_t, default_pads_size>& Pads, const uint32_t Group,
                             const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions) {
  dnn_lib::inlining::fwdLibConvolution3DInst<out0Type, in2Type>(out0, in0, in1, in2, Kernels, Strides, Pads, Group,
                                                                flags, minionOffset, assignedMinions);
}

////////////////////////////////////////////////////////////////////////////////
// Template specializations (declared with 'extern template' in LibNodes.h)
////////////////////////////////////////////////////////////////////////////////
template void fwdLibConvolution3DInst<FloatTy, FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2,
                                                        const std::array<uint32_t, default_kernels_size>& Kernels,
                                                        const std::array<uint32_t, default_kernels_size>& Strides,
                                                        const std::array<uint32_t, default_pads_size>& Pads,
                                                        const uint32_t Group, const uint64_t flags,
                                                        const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Float16Ty, Float16Ty>(
  LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2,
  const std::array<uint32_t, default_kernels_size>& Kernels, const std::array<uint32_t, default_kernels_size>& Strides,
  const std::array<uint32_t, default_pads_size>& Pads, const uint32_t Group, const uint64_t flags,
  const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int8QTy, Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2,
                                                        const std::array<uint32_t, default_kernels_size>& Kernels,
                                                        const std::array<uint32_t, default_kernels_size>& Strides,
                                                        const std::array<uint32_t, default_pads_size>& Pads,
                                                        const uint32_t Group, const uint64_t flags,
                                                        const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int8QTy, Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1,
                                                         LibTensor* in2,
                                                         const std::array<uint32_t, default_kernels_size>& Kernels,
                                                         const std::array<uint32_t, default_kernels_size>& Strides,
                                                         const std::array<uint32_t, default_pads_size>& Pads,
                                                         const uint32_t Group, const uint64_t flags,
                                                         const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int16QTy, Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1,
                                                          LibTensor* in2,
                                                          const std::array<uint32_t, default_kernels_size>& Kernels,
                                                          const std::array<uint32_t, default_kernels_size>& Strides,
                                                          const std::array<uint32_t, default_pads_size>& Pads,
                                                          const uint32_t Group, const uint64_t flags,
                                                          const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibConvolution3DInst<Int16QTy, Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1,
                                                          LibTensor* in2,
                                                          const std::array<uint32_t, default_kernels_size>& Kernels,
                                                          const std::array<uint32_t, default_kernels_size>& Strides,
                                                          const std::array<uint32_t, default_pads_size>& Pads,
                                                          const uint32_t Group, const uint64_t flags,
                                                          const uint32_t minionOffset, const uint32_t assignedMinions);
} // namespace dnn_lib
