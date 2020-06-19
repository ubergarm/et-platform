
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibInsertTensorInst(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibInsertTensorInst<in0Type>(out0, in0, Offsets, Count, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibInsertTensorInstThreaded(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibInsertTensorInstThreaded<in0Type>(out0, in0, Offsets, Count, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibInsertTensorInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibInsertTensorInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_array_t & Offsets, const uint32_t Count, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
