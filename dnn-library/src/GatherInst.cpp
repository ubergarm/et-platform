
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibGatherInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibGatherInst<in0Type, in1Type>(out0, in0, in1, BatchDims, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibGatherInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibGatherInstThreaded<in0Type, in1Type>(out0, in0, in1, BatchDims, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibGatherInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInstThreaded<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInstThreaded<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInstThreaded<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInstThreaded<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInstThreaded<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInstThreaded<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInstThreaded<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInstThreaded<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInstThreaded<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInstThreaded<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
