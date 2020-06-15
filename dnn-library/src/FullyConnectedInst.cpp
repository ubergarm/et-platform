
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibFullyConnectedInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFullyConnectedInst<Int8QTy>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibFullyConnectedInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFullyConnectedInst"Threaded"<Int8QTy>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibFullyConnectedInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFullyConnectedInst"Vectorized"<Int8QTy>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibFullyConnectedInst<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst"Threaded"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst"Vectorized"<Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, LibTensor* out2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
