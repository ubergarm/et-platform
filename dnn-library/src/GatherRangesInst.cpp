
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type, ElemKind in1Type>
  void fwdLibGatherRangesInst(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibGatherRangesInst<in0Type, in1Type>(out0, out1, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibGatherRangesInst<FloatTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Float16Ty,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<UInt8QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int16QTy,Int32ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<FloatTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Float16Ty,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<UInt8QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibGatherRangesInst<Int16QTy,Int64ITy>(LibTensor* out0, LibTensor* out1, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
