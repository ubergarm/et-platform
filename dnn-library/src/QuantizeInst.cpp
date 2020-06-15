
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type>
  void fwdLibQuantizeInst(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibQuantizeInst<Int8QTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type>
  void fwdLibQuantizeInst"Threaded"(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibQuantizeInst"Threaded"<Int8QTy>(out0, in0, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibQuantizeInst<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst<UInt8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst<Int32QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst"Threaded"<UInt8QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst"Threaded"<Int16QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibQuantizeInst"Threaded"<Int32QTy>(LibTensor* out0, LibTensor* out0, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
