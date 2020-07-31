
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibResizeBilinearInst(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibResizeBilinearInst<out0Type>(out0, in0, RszScale, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibResizeBilinearInst<FloatTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Int32QTy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibResizeBilinearInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const std::array<float, default_rszscale_size> & RszScale, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
