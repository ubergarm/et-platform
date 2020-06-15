
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibGatherRangesInst(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibGatherRangesInst<FloatTy,Int32ITy>(out0, out1, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibGatherRangesInst"Threaded"(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibGatherRangesInst"Threaded"<FloatTy,Int32ITy>(out0, out1, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibGatherRangesInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst"Threaded"<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst"Threaded"<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst"Threaded"<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst"Threaded"<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst"Threaded"<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst"Threaded"<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst"Threaded"<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst"Threaded"<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst"Threaded"<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst"Threaded"<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
