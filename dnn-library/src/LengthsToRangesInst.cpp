
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibLengthsToRangesInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLengthsToRangesInst<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibLengthsToRangesInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibLengthsToRangesInst"Threaded"<FloatTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibLengthsToRangesInst<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibLengthsToRangesInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
