
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibBatchedReduceAddInst(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceAddInst<FloatTy>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibBatchedReduceAddInstThreaded(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceAddInstThreaded<FloatTy>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibBatchedReduceAddInstInt8(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceAddInstInt8<FloatTy>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibBatchedReduceAddInstInt8Threaded(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceAddInstInt8Threaded<FloatTy>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibBatchedReduceAddInst<FloatTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Float16Ty>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int8QTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int64ITy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int32ITy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int16QTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8<FloatTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8<Float16Ty>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8<Int8QTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8<Int64ITy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8<Int32ITy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8<Int16QTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8Threaded<FloatTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8Threaded<Float16Ty>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8Threaded<Int8QTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8Threaded<Int64ITy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8Threaded<Int32ITy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInstInt8Threaded<Int16QTy>(LibTensor* out0, LibTensor* in0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
