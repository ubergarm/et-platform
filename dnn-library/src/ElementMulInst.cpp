
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementMulInst(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementMulInst<out0Type, in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementMulInstThreaded(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementMulInstThreaded<out0Type, in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementMulInstVectorized(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementMulInstVectorized<out0Type, in0Type, in1Type>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementMulInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInst<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInst<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInstThreaded<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInstThreaded<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInstThreaded<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInstThreaded<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInstThreaded<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInstVectorized<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInstVectorized<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInstVectorized<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInstVectorized<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementMulInstVectorized<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* in0, LibTensor* in1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
