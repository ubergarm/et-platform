
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibBatchOneHotInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchOneHotInst<FloatTy>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibBatchOneHotInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibBatchOneHotInst"Threaded"<FloatTy>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibBatchOneHotInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibBatchOneHotInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
