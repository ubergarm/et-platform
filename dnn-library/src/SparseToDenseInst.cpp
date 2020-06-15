
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind in0Type>
  void fwdLibSparseToDenseInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseToDenseInst<FloatTy>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSparseToDenseInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseToDenseInst"Threaded"<FloatTy>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind in0Type>
  void fwdLibSparseToDenseInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibSparseToDenseInst"Vectorized"<FloatTy>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibSparseToDenseInst<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Threaded"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Threaded"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Threaded"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Threaded"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Vectorized"<FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Vectorized"<Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Vectorized"<Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Vectorized"<Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibSparseToDenseInst"Vectorized"<Int16QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
