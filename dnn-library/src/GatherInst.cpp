
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibGatherInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibGatherInst<FloatTy,Int32ITy>(out0, in0, in1, BatchDims, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibGatherInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibGatherInst"Threaded"<FloatTy,Int32ITy>(out0, in0, in1, BatchDims, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibGatherInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst"Threaded"<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst"Threaded"<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst"Threaded"<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst"Threaded"<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst"Threaded"<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst"Threaded"<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst"Threaded"<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst"Threaded"<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst"Threaded"<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherInst"Threaded"<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, uint32_t BatchDims, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
