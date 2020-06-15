
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibArgMaxInst(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibArgMaxInst<FloatTy>(out0, in0, Axis, KeepDims, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibArgMaxInst<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
