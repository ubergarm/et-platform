
#include "LibNodes.h"
 
namespace dnn_lib {
  ////////////////////////////////////////////////////////////////////////////////
  // Forward call to corresponding dnn_lib::inlining implementations
  ////////////////////////////////////////////////////////////////////////////////
 
  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementAddInst(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementAddInst<FloatTy,FloatTy,FloatTy>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementAddInst"Threaded"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementAddInst"Threaded"<FloatTy,FloatTy,FloatTy>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////

  template <ElemKind out0Type, ElemKind in0Type, ElemKind in1Type>
  void fwdLibElementAddInst"Vectorized"(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions)
  {
    dnn_lib::inlining::fwdLibElementAddInst"Vectorized"<FloatTy,FloatTy,FloatTy>(out0, in0, in1, flags, minionOffset, assignedMinions);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Template specializations (declared with 'extern template' in LibNodes.h)
  ////////////////////////////////////////////////////////////////////////////////
template void fwdLibElementAddInst<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst"Threaded"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst"Threaded"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst"Threaded"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst"Threaded"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst"Threaded"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst"Vectorized"<FloatTy,FloatTy,FloatTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst"Vectorized"<Float16Ty,Float16Ty,Float16Ty>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst"Vectorized"<Int8QTy,Int8QTy,Int8QTy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst"Vectorized"<Int32ITy,Int32ITy,Int32ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
template void fwdLibElementAddInst"Vectorized"<Int64ITy,Int64ITy,Int64ITy>(LibTensor* out0, LibTensor* out0, LibTensor* out1, const uint64_t flags, const uint32_t minionOffset, const uint32_t assignedMinions);
} // dnn_lib
