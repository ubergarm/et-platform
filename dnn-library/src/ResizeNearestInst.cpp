
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibResizeNearestInst(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibResizeNearestInst<out0Type>(out0, in0, RszScale, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibResizeNearestInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Int32QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeNearestInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
