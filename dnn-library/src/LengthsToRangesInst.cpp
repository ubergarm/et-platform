
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibLengthsToRangesInst(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLengthsToRangesInst<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibLengthsToRangesInstThreaded(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLengthsToRangesInstThreaded<in0Type>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibLengthsToRangesInst<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInstThreaded<FloatTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInstThreaded<Float16Ty>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInstThreaded<Int8QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInstThreaded<Int64ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInstThreaded<Int32ITy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInstThreaded<Int16QTy>(LibTensor* out0, LibTensor* in0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
