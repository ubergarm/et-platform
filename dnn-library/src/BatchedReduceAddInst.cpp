
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibBatchedReduceAddInst(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceAddInst<FloatTy>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibBatchedReduceAddInst"Threaded"(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceAddInst"Threaded"<FloatTy>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibBatchedReduceAddInst"Int8"(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceAddInst"Int8"<FloatTy>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibBatchedReduceAddInst"Int8Threaded"(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchedReduceAddInst"Int8Threaded"<FloatTy>(out0, in0, Axis, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibBatchedReduceAddInst<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8"<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8"<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8"<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8"<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8"<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8"<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchedReduceAddInst"Int8Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, dim_t Axis, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
