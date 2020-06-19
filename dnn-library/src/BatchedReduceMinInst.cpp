
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibBatchedReduceMinInst(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceMinInst<in0Type>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibBatchedReduceMinInstThreaded(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceMinInstThreaded<in0Type>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibBatchedReduceMinInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceMinInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
