
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in2Type>
  void fwdLibFullyConnectedInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibFullyConnectedInst<out0Type, in2Type>(out0, in0, in1, in2, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibFullyConnectedInst<FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Int8QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Int8QTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Int16QTy,Int16QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibFullyConnectedInst<Int16QTy,Int32QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, LibTensor* in2, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
