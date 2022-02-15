
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibBatchedReduceAddInst(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceAddInst<in0Type>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibBatchedReduceAddInst<FloatTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
// FIXME. [SW-11008] at the moment there is no support for those data types  a compile-time error willl happen if used
#if 0
template void fwdLibBatchedReduceAddInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
#endif
template void fwdLibBatchedReduceAddInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // namespace dnn_lib
