
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type>
  void fwdLibArgMaxInst(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibArgMaxInst<out0Type, in0Type>(out0, in0, Axis, KeepDims, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibArgMaxInst<Int64ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int32ITy,FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int64ITy,Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibArgMaxInst<Int32ITy,Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const bool KeepDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
