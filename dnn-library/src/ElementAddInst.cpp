
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementAddInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementAddInst<out0Type, in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementAddInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementAddInstVectorized<out0Type, in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementAddInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInstVectorized<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInstVectorized<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInstVectorized<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInstVectorized<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInstVectorized<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
